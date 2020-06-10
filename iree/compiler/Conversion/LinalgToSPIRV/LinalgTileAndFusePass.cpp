// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgTilingOnBuffers.cpp - Tile and fuse Linalg on Buffers --------===//
//
// Implements a pass to tile and fuse linalg operations on buffers.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse-buffer"

static std::string PromotionMarker = "promotion";

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static constexpr unsigned kMaxWorkgroupRank = 3;

static ArrayRef<int64_t> dropTrailingOnes(ArrayRef<int64_t> vector) {
  if (vector.empty()) return vector;
  auto numTrailingOnes = 0;
  for (unsigned i = vector.size() - 1; i > 0; --i) {
    if (vector[i] != 1) {
      break;
    }
    numTrailingOnes++;
  }
  return vector.drop_back(numTrailingOnes);
}

/// Returns the number of "outer" parallel loops specified in the `linalgOp`.
static unsigned getNumOuterParallelLoops(linalg::LinalgOp linalgOp) {
  if (auto convOp = dyn_cast<linalg::ConvOp>(linalgOp.getOperation())) {
    Optional<DenseIntElementsAttr> padding = convOp.padding();
    if (padding) return convOp.getNumBatchDimensions();
  }
  return linalgOp.iterator_types()
      .getValue()
      .take_while([](Attribute attr) {
        return attr.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      })
      .size();
}

/// Updates the workgroup size used for the dispatch region.
static LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                         ArrayRef<int64_t> workGroupSize) {
  // Need to update both the surrounding FuncOp that has the spv.entry_point_abi
  // attribute, and the hal.executable.
  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body))
    return funcOp.emitError("unhandled dispatch function with multiple blocks");

  if (workGroupSize.size() != 3)
    return funcOp.emitError("expected workgroup size to have three entries");
  SmallVector<int32_t, 3> workGroupSizeVec = llvm::to_vector<3>(llvm::map_range(
      workGroupSize, [](int64_t v) { return static_cast<int32_t>(v); }));

  funcOp.setAttr(
      spirv::getEntryPointABIAttrName(),
      spirv::getEntryPointABIAttr(workGroupSizeVec, funcOp.getContext()));
  return success();
}

namespace {

/// Computes tile sizes (and workgroup size) to use based on operations within
/// the function, and resource constraints on the module.
class TileSizeCalculator {
 public:
  TileSizeCalculator(FuncOp funcOp)
      : resourceLimits(spirv::lookupTargetEnv(funcOp).getResourceLimits()) {
    if (DenseIntElementsAttr attr = spirv::lookupLocalWorkGroupSize(funcOp)) {
      for (auto val : attr.getValues<APInt>())
        workgroupSize.push_back(val.getSExtValue());
    }
    workgroupSize.resize(3, 1);
  }

  /// Compute the tile sizes based on workgroup size specified.
  LogicalResult setTileSizesBasedOnWorkgroupSize(
      ArrayRef<int64_t> vWorkGroupSize) {
    if (!vWorkGroupSize.empty()) {
      vWorkGroupSize = dropTrailingOnes(vWorkGroupSize);
      workgroupSize.assign(vWorkGroupSize.begin(), vWorkGroupSize.end());
      auto rev = reverse(workgroupSize);
      tileSizes.assign(rev.begin(), rev.end());
    }
    return success();
  }

  /// Compute the tile sizes based on the Linalg Ops within the dispatch region.
  LogicalResult setTileSizesBasedOnOps(ArrayRef<linalg::LinalgOp> linalgOps);

  /// Get the current tile size computed.
  ArrayRef<int64_t> getTileSizes() const { return tileSizes; }

  /// Linalg convention is to use 0 for no tiling. If any of the tile dimensions
  /// is set to 1 make it 0.
  SmallVector<int64_t, 3> getTileSizesForLinalg() const {
    return llvm::to_vector<3>(llvm::map_range(
        tileSizes, [](int64_t v) -> int64_t { return v == 1 ? 0 : v; }));
  }

  /// Returns the workgroup size to use based on the tile sizes.
  ArrayRef<int64_t> getWorkGroupSize() const { return workgroupSize; }

 private:
  /// Get the default tile sizes based on just number of dimensions, i.e., "x",
  /// "y", and "z".
  void setTileSizesBasedOnDimensions(unsigned numDims);

  /// Current tile size configuration.
  SmallVector<int64_t, 4> tileSizes;

  /// Workgroup size to use.
  SmallVector<int64_t, 3> workgroupSize;

  /// Attribute for device constraints.
  spirv::ResourceLimitsAttr resourceLimits;
};
}  // namespace

void TileSizeCalculator::setTileSizesBasedOnDimensions(unsigned numDims) {
  tileSizes.clear();
  workgroupSize.clear();
  tileSizes.reserve(3);
  if (numDims == 0) {
    // Scalar case.
    workgroupSize = {1, 1, 1};
    return;
  }
  unsigned maxWorkGroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();

  // Make the tile size 32 along the x-dimension, and then split the remaining
  // maxWorkGroupSize threads amongst the y-dimension or z-dimension.
  unsigned tileSizeX = llvm::PowerOf2Floor(std::min(maxWorkGroupSize, 32u));
  maxWorkGroupSize /= tileSizeX;
  if (numDims == 1) {
    tileSizes = {tileSizeX};
    workgroupSize = {tileSizeX, 1, 1};
    return;
  }
  if (numDims == 2) {
    unsigned tileSizeY = llvm::PowerOf2Floor(maxWorkGroupSize);
    tileSizes = {tileSizeY, tileSizeX};
    workgroupSize = {tileSizeX, tileSizeY, 1};
    return;
  }
  unsigned tileSizeYZ =
      llvm::PowerOf2Floor(static_cast<unsigned>(std::sqrt(maxWorkGroupSize)));
  tileSizes = {tileSizeYZ, tileSizeYZ, tileSizeX};
  workgroupSize = {tileSizeX, tileSizeYZ, tileSizeYZ};
}

LogicalResult TileSizeCalculator::setTileSizesBasedOnOps(
    ArrayRef<linalg::LinalgOp> linalgOps) {
  tileSizes.clear();
  // The tile size will be driven by operations like matmul, conv, etc. within
  // the list. So see what operation exists in the list to decide the tile size.
  // If there are two such operations in the list, return error.
  bool hasMatmul = false;
  unsigned numParallelLoops = kMaxWorkgroupRank;
  for (linalg::LinalgOp op : linalgOps) {
    // If there is no marker on this op (i.e. a marker to prevent tile), add an
    // explicit marker to indicate that the op is to be tiled. Makes subsequent
    // lowering simpler.
    if (isa<linalg::MatmulOp>(op.getOperation())) {
      if (hasMatmul)
        return op.emitError(
            "unhandled multiple matmuls within dispatch region");
      hasMatmul = true;
    }
    numParallelLoops = std::min(numParallelLoops, getNumOuterParallelLoops(op));
  }
  if (hasMatmul) {
    // TODO: For now just hard wire this, but we can do better.
    tileSizes = {8, 8, 4};
    workgroupSize = {8, 8, 1};
    return success();
  }
  setTileSizesBasedOnDimensions(numParallelLoops);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

/// Allocation callback for allocation workgroup local memory.
static Value allocateWorkgroupMemory(OpBuilder &b, SubViewOp subview,
                                     ArrayRef<Value> boundingSubViewSize,
                                     OperationFolder *folder) {
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape(boundingSubViewSize.size(),
                                ShapedType::kDynamicSize);
  return b.create<AllocOp>(
      subview.getLoc(),
      MemRefType::get(shape, subview.getType().getElementType(), {},
                      getWorkgroupMemorySpace()),
      boundingSubViewSize);
}

/// Deallocation callback for allocation workgroup local memory.
static LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  auto allocOp = buffer.getDefiningOp<AllocOp>();
  b.create<DeallocOp>(allocOp.getLoc(), buffer);
  return success();
}

/// Insert barrier after `op`.
static void insertBarrierAfter(OpBuilder &b, Location loc, Operation *op) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(op);
  b.create<spirv::ControlBarrierOp>(loc, spirv::Scope::Workgroup,
                                    spirv::Scope::Workgroup,
                                    spirv::MemorySemantics::AcquireRelease);
}

/// Function used as callback for copyin/copyout in promotion pattern used to
/// promote subviews to workgroup memory.
static LogicalResult copyToFromWorkgroupMemory(
    OpBuilder &b, Value src, Value dst, StringRef marker = PromotionMarker) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, marker);
  return success();
}

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, FunctionPass> {
  LinalgTileAndFusePass(ArrayRef<int64_t> workGroupSize = {},
                        bool useWorkgroupMem = false)
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {
    this->useWorkgroupMemory = useWorkgroupMem;
  }
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass) {}

  void runOnFunction() override;

  Option<bool> useWorkgroupMemory{
      *this, "use-workgroup-memory",
      llvm::cl::desc("Promote subviews to use workgroup memory"),
      llvm::cl::init(false)};

 private:
  SmallVector<int64_t, 3> workGroupSize;
};

/// Pattern to promote subviews to memory.
// TODO(ravishankarm): Generalize this for other operations.
struct PromoteSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteSubviewsPattern(MLIRContext *context,
                         linalg::LinalgPromotionOptions options,
                         linalg::LinalgMarker marker = linalg::LinalgMarker(),
                         PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!hasWorkItemMarker(op)) return failure();
    return linalg::LinalgPromotionPattern<linalg::MatmulOp>::matchAndRewrite(
        op, rewriter);
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();
  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body.getBlocks())) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }
  Block &block = body.front();
  auto linalgOps = block.getOps<linalg::LinalgOp>();
  if (linalgOps.empty()) return;

  TileSizeCalculator tileSizeCalculator(funcOp);
  if (workGroupSize.empty()) {
    // Get the tile sizes to use for the lowering.
    SmallVector<int64_t, 3> tileSizes;
    SmallVector<linalg::LinalgOp, 1> opsVec(linalgOps.begin(), linalgOps.end());
    if (failed(tileSizeCalculator.setTileSizesBasedOnOps(opsVec)))
      return signalPassFailure();
  } else {
    tileSizeCalculator.setTileSizesBasedOnWorkgroupSize(workGroupSize);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# workgroup sizes at start: [";
    interleaveComma(workGroupSize, llvm::dbgs());
    llvm::dbgs() << "]\ntile sizes: [";
    interleaveComma(tileSizeCalculator.getTileSizes(), llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  OwningRewritePatternList tilingPatterns;
  tilingPatterns.insert<linalg::LinalgTilingPattern<linalg::ConvOp>,
                        linalg::LinalgTilingPattern<linalg::MatmulOp>,
                        linalg::LinalgTilingPattern<linalg::PoolingMaxOp>,
                        linalg::LinalgTilingPattern<linalg::PoolingMinOp>,
                        linalg::LinalgTilingPattern<linalg::PoolingSumOp>>(
      context,
      linalg::LinalgTilingOptions()
          .setTileSizes(tileSizeCalculator.getTileSizesForLinalg())
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
      linalg::LinalgMarker(ArrayRef<Identifier>(),
                           Identifier::get(getWorkItemMarker(), context)));
  applyPatternsAndFoldGreedily(getOperation(), tilingPatterns);

  if (useWorkgroupMemory) {
    // The promotion patterns are put separate from the tiling patterns to make
    // sure that the allocated scratchspace memory is constant sizes which
    // requires some folding to trigger.
    OwningRewritePatternList promotionPatterns;
    promotionPatterns.insert<PromoteSubviewsPattern>(
        context,
        linalg::LinalgPromotionOptions()
            .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                          deallocateWorkgroupMemory)
            .setCopyInOutFns(
                [&](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                  return copyToFromWorkgroupMemory(b, src, dst);
                },
                [&](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                  return copyToFromWorkgroupMemory(b, src, dst);
                }),
        linalg::LinalgMarker(Identifier::get(getWorkItemMarker(), context),
                             Identifier::get(PromotionMarker, context)));
    applyPatternsAndFoldGreedily(getOperation(), promotionPatterns);
  }

  // Add barrier after all linalg operations marked with workitem marker.
  OpBuilder builder(context);
  funcOp.walk([&builder](linalg::LinalgOp linalgOp) {
    if (hasMarker(linalgOp, PromotionMarker)) {
      setWorkItemMarker(linalgOp);
      insertBarrierAfter(builder, linalgOp.getLoc(), linalgOp);
    }
  });

  // Update the workgroup size to be consistent with the tile sizes used. Note
  // the tile sizes are ordered from outer most to inner most loops. The
  // heuristic is to map the inner loops to x, the next outer (if it exists) to
  // y, and the next outer (if it exists) to z. So tile sizes are reversed to
  // get the workgroup size.
  if (failed(
          updateWorkGroupSize(funcOp, tileSizeCalculator.getWorkGroupSize())))
    return signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workGroupSize, bool useWorkgroupMemory) {
  return std::make_unique<LinalgTileAndFusePass>(workGroupSize,
                                                 useWorkgroupMemory);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
