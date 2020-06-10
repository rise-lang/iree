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

//===- ConvertToGPUPass.cpp -----------------------------------------------===//
//
// Partition computation within dispatch function to workgroups/workitems.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Loop utilities
//===----------------------------------------------------------------------===//

/// Builds an empty scf.for operation. The default builder adds an entry basic
/// block which needs to be avoided here.
static scf::ForOp buildEmptyForOp(Location loc, OpBuilder &builder, Value lb,
                                  Value ub, Value step) {
  OperationState state(loc, scf::ForOp::getOperationName());
  state.addOperands({lb, ub, step});
  state.addRegion();
  return cast<scf::ForOp>(builder.createOperation(state));
}

/// Builds and empty scf.parallel operation without the entry basic block.
static scf::ParallelOp buildEmptyParallelOp(Location loc, OpBuilder &builder,
                                            ArrayRef<Value> lbs,
                                            ArrayRef<Value> ubs,
                                            ArrayRef<Value> steps) {
  OperationState state(loc, scf::ParallelOp::getOperationName());
  state.addOperands(lbs);
  state.addOperands(ubs);
  state.addOperands(steps);
  state.addRegion();
  return cast<scf::ParallelOp>(builder.createOperation(state));
}

/// Builds and empty scf.if operation without the then and else blocks.
static scf::IfOp buildEmptyIfOp(Location loc, OpBuilder &builder, Value cond) {
  OperationState state(loc, scf::IfOp::getOperationName());
  state.addOperands(cond);
  state.addRegion();
  state.addRegion();
  return cast<scf::IfOp>(builder.createOperation(state));
}

namespace {
struct LoopBounds {
  Value lb;
  Value ub;
  Value step;
};
}  // namespace

/// Replaces a loop.parallelOp with an optional loop.parallel op and nested
/// loop.for operations. To create the loop.parallel op as the outermost loop,
/// pass the lower bound, upper bound and steps in `newPLoopLbs`, `newPLoopUbs`,
/// and `newPLoopStep` respectively. The bounds of the inner loop.for operations
/// to be created are passed in `forLbs`, `forUbs`, and `forStep`. The
/// `permutation` vector contains a mapping from the original loop order, to the
/// loop order to be generated.
static Operation *replacePLoopOp(ConversionPatternRewriter &rewriter,
                                 scf::ParallelOp pLoopOp,
                                 ArrayRef<LoopBounds> newPLoopBounds,
                                 ArrayRef<LoopBounds> forBounds,
                                 ArrayRef<unsigned> permutation) {
  assert(!forBounds.empty() && "unhandled case of no loop.for created");
  unsigned numLoops = pLoopOp.getNumLoops();
  Location loc = pLoopOp.getLoc();
  assert(forBounds.size() + newPLoopBounds.size() == numLoops &&
         "cannot drop loops when splitting loop.parallel operation");
  assert(permutation.size() == numLoops);
  OpBuilder::InsertionGuard guard(rewriter);

  // Need a signature conversion for the body of the loop.parallel operation,
  // before can it can be used as the body of the innermost loop created here.
  TypeConverter::SignatureConversion signatureConverter(numLoops);
  Operation *outermostLoop = nullptr;
  auto permuteIt = permutation.begin();

  // Create the loop.parallel operation as the outermost loop, if specified.
  if (!newPLoopBounds.empty()) {
    auto lbs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) -> Value { return bounds.lb; }));
    auto ubs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.ub; }));
    auto steps = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.step; }));
    auto newPLoop = rewriter.create<scf::ParallelOp>(loc, lbs, ubs, steps);
    for (auto iv : newPLoop.getInductionVars()) {
      signatureConverter.remapInput(*permuteIt, iv);
      permuteIt++;
    }
    rewriter.setInsertionPointToStart(newPLoop.getBody());
    outermostLoop = newPLoop.getOperation();
  }

  // Generate the nested loop.for operations with the bounds passed.
  for (auto it : enumerate(forBounds)) {
    Value lb = it.value().lb, ub = it.value().ub, step = it.value().step;
    if (it.index() != forBounds.size() - 1) {
      auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.remapInput(*permuteIt, forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    } else {
      // For the last loop, move the body of the loop.parallel op as the body of
      // the loop after signature conversion.
      auto forOp = buildEmptyForOp(loc, rewriter, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.addInputs(*permuteIt, rewriter.getIndexType());
      Region &pLoopOpRegion = pLoopOp.getLoopBody();
      rewriter.applySignatureConversion(&pLoopOpRegion, signatureConverter);
      Region &forOpRegion = forOp.getLoopBody();
      rewriter.inlineRegionBefore(pLoopOpRegion, forOpRegion,
                                  forOpRegion.begin());
    }
    permuteIt++;
  }
  rewriter.eraseOp(pLoopOp);
  return outermostLoop;
}

/// Serializes the dimensions of the loop.parallel specified in
/// `serializedDimensions`, by creating an nested loop.for operation for each
/// dimension.
// TODO(ravishankarm): Move this into LoopUtils.h in MLIR.
static Operation *serializeDimensions(ConversionPatternRewriter &rewriter,
                                      scf::ParallelOp pLoopOp,
                                      ArrayRef<unsigned> serializedDimensions) {
  assert(!serializedDimensions.empty() &&
         "unhandled corner case of no serializing dims");
  OpBuilder::InsertionGuard guard(rewriter);
  DenseSet<unsigned> serializedDimSet;
  serializedDimSet.insert(serializedDimensions.begin(),
                          serializedDimensions.end());
  assert(serializedDimSet.size() == serializedDimensions.size() &&
         "cannot repeat dimensions during serialization of loop.parallel");
  SmallVector<LoopBounds, 2> newPLoopBounds, forBounds;
  SmallVector<unsigned, 2> permutation;
  auto lbs = pLoopOp.lowerBound();
  auto ubs = pLoopOp.upperBound();
  auto steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, pLoopOp.getNumLoops())) {
    if (serializedDimSet.count(i)) {
      forBounds.push_back({lbs[i], ubs[i], steps[i]});
    } else {
      newPLoopBounds.push_back({lbs[i], ubs[i], steps[i]});
      permutation.push_back(i);
    }
  }
  permutation.append(serializedDimensions.begin(), serializedDimensions.end());
  return replacePLoopOp(rewriter, pLoopOp, newPLoopBounds, forBounds,
                        permutation);
}

/// Serialize all inner dimensions of a `pLoopOp` starting from `serializeFrom`.
static Operation *serializeDimensionsFrom(ConversionPatternRewriter &rewriter,
                                          scf::ParallelOp pLoopOp,
                                          unsigned serializeFrom) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(serializeFrom > 0 && "unhandled serializaing all dimensions");
  assert(serializeFrom < numLoops &&
         "unhandled corner case of no serialization");
  SmallVector<unsigned, 2> serializedDimensions;
  for (unsigned dim : llvm::seq(serializeFrom, numLoops))
    serializedDimensions.push_back(dim);
  return serializeDimensions(rewriter, pLoopOp, serializedDimensions);
}

/// Collapse all loops in a scf.parallel into one scf.parallel operation. This
/// is done by
/// 1) Normalize the loop bounds to be [0, (ub - lb) / step)
/// 2) Compute the total number of iterations.
/// 3) From the induction variable of the modified loop, compute the values of
///     the original induction variables by dileanirization.
scf::ParallelOp collapseParallelLoops(ConversionPatternRewriter &rewriter,
                                      scf::ParallelOp pLoopOp) {
  // Do not handle reductions.
  if (pLoopOp.getNumReductions()) return nullptr;

  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops == 1) return pLoopOp;

  // Compute the number of iterations of each loops starting from the innermost
  Location loc = pLoopOp.getLoc();
  Value totalNumIterations = rewriter.create<ConstantIndexOp>(loc, 1);

  // Track the "stride" of each loop, i.e. product of the total number of
  // iterations of the inner loops.
  SmallVector<Value, 2> iterationStride;
  iterationStride.resize(pLoopOp.getNumLoops());
  auto lbs = pLoopOp.lowerBound();
  auto ubs = pLoopOp.upperBound();
  auto steps = pLoopOp.step();
  for (unsigned i = numLoops; i != 0; --i) {
    Value lb = lbs[i - 1], ub = ubs[i - 1], step = steps[i - 1];
    Value iterCount = rewriter.create<SignedDivIOp>(
        loc, rewriter.create<SubIOp>(loc, ub, lb), step);
    iterationStride[i - 1] = totalNumIterations;
    totalNumIterations =
        rewriter.create<MulIOp>(loc, totalNumIterations, iterCount);
  }
  Value newLb = rewriter.create<ConstantIndexOp>(loc, 0);
  Value newStep = rewriter.create<ConstantIndexOp>(loc, 1);
  scf::ParallelOp newPLoopOp =
      rewriter.create<scf::ParallelOp>(loc, newLb, totalNumIterations, newStep);
  OpBuilder::InsertionGuard guard(rewriter);
  Block &pLoopBody = pLoopOp.getLoopBody().front();
  rewriter.setInsertionPointToStart(&newPLoopOp.getLoopBody().front());
  Value loopIv = *newPLoopOp.getInductionVars().begin();
  BlockAndValueMapping map;
  for (unsigned i : llvm::seq<unsigned>(0, numLoops)) {
    Value iterNum =
        rewriter.create<SignedDivIOp>(loc, loopIv, iterationStride[i]);
    Value newIv = rewriter.create<AddIOp>(
        loc, lbs[i], rewriter.create<MulIOp>(loc, iterNum, steps[i]));
    map.map(pLoopBody.getArgument(i), newIv);
    loopIv = rewriter.create<SignedRemIOp>(loc, loopIv, iterationStride[i]);
  }
  for (Operation &op : pLoopBody) {
    if (isa<scf::YieldOp>(op)) continue;
    rewriter.clone(op, map);
  }
  rewriter.eraseOp(pLoopOp);
  return newPLoopOp;
}

//===----------------------------------------------------------------------===//
// GPU processor ID mapping utilities
//===----------------------------------------------------------------------===//

/// Distribute loop.parallel to processors with the processors logically
/// arranged with same dimensionality as the number of loops, i.e. a
/// loop.parallel with 2 loops to a 2D grid of processors. `processorIDs` and
/// `numProcessors` must be of same size as the number of loops and are the
/// values to use for process ID and number of processors along each dimension
/// in the distributed code.
/// This method accounts for the case where the number of processors is not
/// enough to execute the entire iteration space with one iteration mapped to
/// each processor. So implements a block-cyclic distribution with eaach block
/// size being equal to bumber of processors.
static LogicalResult mapToProcessors(ConversionPatternRewriter &rewriter,
                                     scf::ParallelOp pLoopOp,
                                     ArrayRef<Value> processorIDs,
                                     ArrayRef<Value> numProcessors) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(numLoops == processorIDs.size() &&
         "expected as many ids as number of loops");
  assert(numLoops == numProcessors.size() &&
         "expected as many nprocs as number of loops");
  SmallVector<LoopBounds, 2> forBounds;
  SmallVector<unsigned, 2> permutation;
  forBounds.reserve(numLoops);
  permutation.reserve(numLoops);
  Location loc = pLoopOp.getLoc();
  auto lbs = pLoopOp.lowerBound(), ubs = pLoopOp.upperBound(),
       steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, processorIDs.size())) {
    Value mappedLb = rewriter.create<AddIOp>(
        loc, lbs[i], rewriter.create<MulIOp>(loc, steps[i], processorIDs[i]));
    Value mappedStep = rewriter.create<MulIOp>(loc, steps[i], numProcessors[i]);
    forBounds.push_back({mappedLb, ubs[i], mappedStep});
    permutation.push_back(i);
  }
  replacePLoopOp(rewriter, pLoopOp, /*newPLoopBounds=*/{}, forBounds,
                 permutation);
  return success();
}

/// Distribute loop.parallel to processors with the processors logically
/// arranged with same dimensionality as the number of loops, i.e. a
/// loop.parallel with 2 loops to a 2D grid of processors. `processorIDs` and
/// `numProcessors` must be of same size as the number of loops and are the
/// values to use for process ID and number of processors along each dimension
/// in the distributed code.
/// This method assumes that the number of processors is greater than or equal
/// to the number of iterations. So just generates an if statement to mask of
/// processors with no work.
static LogicalResult mapToProcessorsAndGuard(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp,
    ArrayRef<Value> processorIDs) {
  unsigned numLoops = pLoopOp.getNumLoops();
  Location loc = pLoopOp.getLoc();
  assert(numLoops == processorIDs.size() &&
         "expected as many ids as number of loops");
  Value cond = nullptr;
  TypeConverter::SignatureConversion signatureConverter(numLoops);
  auto ubs = pLoopOp.upperBound();
  for (unsigned i : llvm::seq<unsigned>(0, numLoops)) {
    Value cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt,
                                        processorIDs[i], ubs[i]);
    cond = (cond ? rewriter.create<AndOp>(loc, cond, cmp) : cmp);
    signatureConverter.remapInput(i, processorIDs[i]);
  }
  scf::IfOp ifOp = buildEmptyIfOp(loc, rewriter, cond);
  Region &pLoopOpRegion = pLoopOp.getLoopBody();
  rewriter.applySignatureConversion(&pLoopOpRegion, signatureConverter);
  Region &ifOpRegion = ifOp.getRegion(0);
  rewriter.inlineRegionBefore(pLoopOpRegion, ifOpRegion, ifOpRegion.begin());
  rewriter.eraseOp(pLoopOp);
  return success();
}

namespace {
struct ProcessorIdAndCount {
  Value id;
  Value count;
};

/// These are class declarations that are only used for template
/// specialization. They wont be needed if GPU dialect has ops for global
/// invocation ID directly.
class GPUGlobalId;
class GPUGlobalCount;
}  // namespace

template <typename GPUIdOp, typename GPUCountOp>
static ProcessorIdAndCount getGPUProcessorIdAndCount(
    Location loc, StringRef dim, ConversionPatternRewriter &rewriter) {
  Type indexType = rewriter.getIndexType();
  return {
      rewriter.create<GPUIdOp>(loc, indexType, rewriter.getStringAttr(dim)),
      rewriter.create<GPUCountOp>(loc, indexType, rewriter.getStringAttr(dim))};
}

template <>
ProcessorIdAndCount getGPUProcessorIdAndCount<GPUGlobalId, GPUGlobalCount>(
    Location loc, StringRef dim, ConversionPatternRewriter &rewriter) {
  Type indexType = rewriter.getIndexType();
  Value gridDim = rewriter.create<gpu::GridDimOp>(loc, indexType,
                                                  rewriter.getStringAttr(dim));
  Value blockId = rewriter.create<gpu::BlockIdOp>(loc, indexType,
                                                  rewriter.getStringAttr(dim));
  Value blockDim = rewriter.create<gpu::BlockDimOp>(
      loc, indexType, rewriter.getStringAttr(dim));
  Value threadId = rewriter.create<gpu::ThreadIdOp>(
      loc, indexType, rewriter.getStringAttr(dim));
  return {rewriter.create<AddIOp>(
              loc, rewriter.create<MulIOp>(loc, blockId, blockDim), threadId),
          rewriter.create<MulIOp>(loc, blockDim, gridDim)};
}

template <typename GPUIdOp, typename GPUCountOp>
static void getGPUProcessorIdsAndCounts(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        unsigned numDims,
                                        MutableArrayRef<Value> id,
                                        MutableArrayRef<Value> count) {
  ArrayRef<StringRef> dims = {"x", "y", "z"};
  assert(id.size() == numDims);
  assert(count.size() == numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    ProcessorIdAndCount idAndCount =
        getGPUProcessorIdAndCount<GPUIdOp, GPUCountOp>(loc, dims[i], rewriter);
    id[numDims - 1 - i] = idAndCount.id;
    count[numDims - 1 - i] = idAndCount.count;
  }
}

/// Distribute loop.parallel to processors where `IdOp` is used to get the
/// processor ID and `DimOp` is used to get the number of processors along a
/// dimension.
template <typename GPUIdOp, typename GPUCountOp>
static LogicalResult mapToProcessor(ConversionPatternRewriter &rewriter,
                                    scf::ParallelOp pLoopOp) {
  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops > 3) {
    pLoopOp =
        cast<scf::ParallelOp>(serializeDimensionsFrom(rewriter, pLoopOp, 3));
    numLoops = 3;
  }
  SmallVector<Value, 2> id(numLoops), count(numLoops);
  getGPUProcessorIdsAndCounts<GPUIdOp, GPUCountOp>(pLoopOp.getLoc(), rewriter,
                                                   numLoops, id, count);
  return mapToProcessors(rewriter, pLoopOp, id, count);
}

/// Distribute loop.parallel to processors where `IdOp` is used to get the
/// processor ID and `DimOp` is used to get the number of processors along a
/// dimension. Assumes that the number of processors will be less than equal to
/// the number of iterations of the pLoopOp along all dimensions.
template <typename GPUIdOp, typename GPUCountOp>
static LogicalResult mapToProcessorsAndGuard(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp) {
  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops > 3) {
    pLoopOp =
        cast<scf::ParallelOp>(serializeDimensionsFrom(rewriter, pLoopOp, 3));
    numLoops = 3;
  }
  SmallVector<Value, 2> id(numLoops), count(numLoops);
  getGPUProcessorIdsAndCounts<GPUIdOp, GPUCountOp>(pLoopOp.getLoc(), rewriter,
                                                   numLoops, id, count);
  return mapToProcessorsAndGuard(rewriter, pLoopOp, id);
}

/// Distribute the loop.parallel to workgroups.
static LogicalResult mapToWorkgroups(ConversionPatternRewriter &rewriter,
                                     scf::ParallelOp pLoopOp) {
  return mapToProcessor<gpu::BlockIdOp, gpu::GridDimOp>(rewriter, pLoopOp);
}

/// Distribute loop.parallel to workitems using local invocation ID.
static LogicalResult mapToLocalInvocationId(ConversionPatternRewriter &rewriter,
                                            scf::ParallelOp pLoopOp) {
  return mapToProcessorsAndGuard<gpu::ThreadIdOp, gpu::BlockDimOp>(rewriter,
                                                                   pLoopOp);
}

/// Distribute loop.parallel to workitems using global invocation ID. The GPU
/// dialect doesn't have a direct operation to do this. This could be done using
/// id = blockIdx * blockDim + gridIdx. count = blockDim * gridDim.
static LogicalResult mapToGlobalInvocationId(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp) {
  return mapToProcessorsAndGuard<GPUGlobalId, GPUGlobalCount>(rewriter,
                                                              pLoopOp);
}

//===----------------------------------------------------------------------===//
// Pass and patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Pass to convert from tiled and fused linalg ops into gpu.func.
struct ConvertToGPUPass : public PassWrapper<ConvertToGPUPass, FunctionPass> {
  void runOnFunction() override;
};

/// Pattern to map loop.parallel to workgroups.
struct PartitionPLoopToWorkgroups
    : public OpConversionPattern<scf::ParallelOp> {
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      scf::ParallelOp pLoopOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    return mapToWorkgroups(rewriter, pLoopOp);
  }
};

/// Map tiled linalg op to workitems by lowering it to loop.parallel and
/// partitioning it to workitems.
template <typename LinalgOpTy>
struct MapLinalgOpToLocalInvocationId : public OpConversionPattern<LinalgOpTy> {
  using OpConversionPattern<LinalgOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LinalgOpTy linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Check for marker that specifies that the linalg op is to be partitioned
    // across threads within a workgroup.
    if (!hasWorkItemMarker(linalgOp)) return failure();
    Optional<linalg::LinalgLoops> loops =
        linalg::linalgLowerOpToLoops<scf::ParallelOp, LinalgOpTy>(rewriter,
                                                                  linalgOp);
    if (!loops) return failure();
    if (!loops.getValue().empty()) {
      scf::ParallelOp pLoopOp = dyn_cast<scf::ParallelOp>(loops.getValue()[0]);
      if (!pLoopOp || failed(mapToLocalInvocationId(rewriter, pLoopOp)))
        return failure();
    }
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

/// Map linalg operation to execute on GPU in parallel by mapping the parallel
/// loops to "GlobalInvocationId".
template <typename LinalgOpTy>
struct MapLinalgOpToGlobalInvocationId
    : public OpConversionPattern<LinalgOpTy> {
  using OpConversionPattern<LinalgOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LinalgOpTy linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If marker exists do nothing.
    if (hasMarker(linalgOp)) return failure();
    Optional<linalg::LinalgLoops> loops =
        linalg::linalgLowerOpToLoops<scf::ParallelOp, LinalgOpTy>(rewriter,
                                                                  linalgOp);
    if (!loops) return failure();

    SmallVector<int64_t, 3> workgroupSize(3, 1);
    if (!loops.getValue().empty()) {
      scf::ParallelOp pLoopOp = dyn_cast<scf::ParallelOp>(loops.getValue()[0]);
      // If there are parallel loops partition them to threads using global
      // invocation ID.
      if (pLoopOp) {
        std::vector<unsigned> collapsedDims;
        for (unsigned i : llvm::seq<unsigned>(0, pLoopOp.getNumLoops()))
          collapsedDims.push_back(i);
        pLoopOp = collapseParallelLoops(rewriter, pLoopOp);
        if (!pLoopOp) return failure();
        if (failed(mapToGlobalInvocationId(rewriter, pLoopOp)))
          return failure();
        workgroupSize = {32, 1, 1};
      }
    }
    rewriter.eraseOp(linalgOp);
    FuncOp funcOp = linalgOp.template getParentOfType<FuncOp>();
    if (funcOp) {
      updateWorkGroupSize(funcOp, workgroupSize);
    }
    return success();
  }
};

/// Remove the linalg.range operation created when lowering to loops.
struct RemoveLinalgRange : public OpConversionPattern<linalg::RangeOp> {
  using OpConversionPattern<linalg::RangeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      linalg::RangeOp rangeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!rangeOp.getResult().use_empty()) return failure();
    rewriter.eraseOp(rangeOp);
    return success();
  }
};
}  // namespace

void ConvertToGPUPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body)) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  // After this pass Linalg and loop.parallel ops should be gone.
  target.addIllegalOp<scf::ParallelOp>();
  target.addIllegalDialect<linalg::LinalgDialect>();
  // Reshape ops are treated legal since they just change the way the underlying
  // buffer is viewed. These are legalized downstream. They become no ops when
  // lowering to SPIR-V since the SPIR-V code uses linearized arrays.
  target.addLegalOp<linalg::ReshapeOp>();
  // Let the rest fall through.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  OwningRewritePatternList patterns;

  patterns.insert<
  // clang-format off
#define ADD_ALL_LINALG_PATTERNS(OP_NAME)                                       \
  MapLinalgOpToGlobalInvocationId<OP_NAME>,                                    \
  MapLinalgOpToLocalInvocationId<OP_NAME>

    ADD_ALL_LINALG_PATTERNS(linalg::ConvOp),
    ADD_ALL_LINALG_PATTERNS(linalg::CopyOp),
    ADD_ALL_LINALG_PATTERNS(linalg::FillOp),
    ADD_ALL_LINALG_PATTERNS(linalg::GenericOp),
    ADD_ALL_LINALG_PATTERNS(linalg::IndexedGenericOp),
    ADD_ALL_LINALG_PATTERNS(linalg::MatmulOp),
    ADD_ALL_LINALG_PATTERNS(linalg::PoolingMaxOp),
    ADD_ALL_LINALG_PATTERNS(linalg::PoolingMinOp),
    ADD_ALL_LINALG_PATTERNS(linalg::PoolingSumOp),

#undef ADD_ALL_LINALG_PATTERNS

    PartitionPLoopToWorkgroups, RemoveLinalgRange>(context);
  // clang-format on

  if (failed(applyFullConversion(funcOp, target, patterns)))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> createConvertToGPUPass() {
  return std::make_unique<ConvertToGPUPass>();
}

static PassRegistration<ConvertToGPUPass> pass(
    "iree-codegen-convert-to-gpu", "Map tiled linalg and loop ops to GPU",
    [] { return std::make_unique<ConvertToGPUPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
