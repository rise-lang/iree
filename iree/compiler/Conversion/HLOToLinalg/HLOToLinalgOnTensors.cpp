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

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/xla_legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
static ArrayAttr getParallelAndReductionIterAttrs(Builder b, unsigned nLoops,
                                                  unsigned nReduction) {
  SmallVector<Attribute, 3> attrs(nLoops - nReduction,
                                  b.getStringAttr("parallel"));
  attrs.append(nReduction, b.getStringAttr("reduction"));
  return b.getArrayAttr(attrs);
}

ShapedType getXLAOpResultType(Operation* op) {
  return op->getResult(0).getType().cast<ShapedType>();
}

template <bool isLHLO = true>
bool verifyXLAOpTensorSemantics(Operation* op) {
  auto verifyType = [&](Value val) -> bool {
    return (val.getType().isa<RankedTensorType>());
  };
  return llvm::all_of(op->getOperands(), verifyType) &&
         llvm::all_of(op->getResults(), verifyType);
}

/// Conversion pattern for splat constants that are not zero-dim tensors, i.e
/// constant dense<...> : tensor<?xelem-type> -> linalg.generic op.
class SplatConstConverter : public OpConversionPattern<ConstantOp> {
 public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!verifyXLAOpTensorSemantics(op)) {
      return failure();
    }
    auto resultType = getXLAOpResultType(op);
    if (resultType.getRank() == 0) return failure();
    auto valueAttr = op.value().template cast<DenseElementsAttr>();
    if (!valueAttr.isSplat()) return failure();

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    auto nloops = std::max<unsigned>(resultType.getRank(), 1);
    auto loc = op.getLoc();

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, args, rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(1),
        rewriter.getAffineMapArrayAttr(rewriter.getMultiDimIdentityMap(nloops)),
        getParallelAndReductionIterAttrs(rewriter, nloops, /*nReduction=*/0),
        /*doc=*/nullptr,
        /*library_call=*/nullptr);
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    rewriter.setInsertionPointToEnd(block);
    auto stdConstOp =
        rewriter.create<mlir::ConstantOp>(loc, valueAttr.getSplatValue());
    rewriter.create<linalg::YieldOp>(loc, stdConstOp.getResult());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

class ConcatenateConverter
    : public OpConversionPattern<xla_hlo::ConcatenateOp> {
 public:
  using OpConversionPattern<xla_hlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_hlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();
    int dim = op.dimension().getSExtValue();
    int rank = args[0].getType().cast<ShapedType>().getRank();

    SmallVector<Attribute, 2> indexingMaps;
    SmallVector<AffineExpr, 4> exprs;
    exprs.resize(rank);
    for (int i = 0, e = rank - 1; i < e; ++i)
      exprs[i] = rewriter.getAffineDimExpr(i);

    // [0, `rank`-1) dims are mapping to non-concatenated dimensions. `rank-1`th
    // is for the `dim` dimension of result, `rank`th is for args[0], and so on.
    int nloops = rank + args.size();
    for (int i = 0, e = args.size(); i < e; ++i) {
      exprs[rank - 1] = rewriter.getAffineDimExpr(rank + i);
      indexingMaps.emplace_back(AffineMapAttr::get(AffineMap::get(
          nloops, /*symbolCount=*/0, exprs, rewriter.getContext())));
    }
    exprs[rank - 1] = rewriter.getAffineDimExpr(rank - 1);
    indexingMaps.emplace_back(AffineMapAttr::get(AffineMap::get(
        nloops, /*symbolCount=*/0, exprs, rewriter.getContext())));

    SmallVector<Type, 4> bodyArgTypes, opResultTypes;
    auto resultType = op.getResult().getType().dyn_cast<ShapedType>();
    if (!resultType) return failure();
    opResultTypes.push_back(resultType);
    // Also make the dimension to be concatenated not a parallel loop.
    int nonParallelLoops = nloops - rank + 1;
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, opResultTypes, args,
        rewriter.getI64IntegerAttr(args.size()),           // args_in
        rewriter.getI64IntegerAttr(opResultTypes.size()),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        getParallelAndReductionIterAttrs(rewriter, nloops, nonParallelLoops),
        /*doc=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    bodyArgTypes.append(nloops, rewriter.getIndexType());
    bodyArgTypes.append(args.size(), resultType.getElementType());
    block->addArguments(bodyArgTypes);
    rewriter.setInsertionPointToEnd(block);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    Value dimArg = block->getArgument(rank - 1);
    Value res = block->getArgument(nloops);
    for (int i = 0, e = args.size(); i < e; ++i) {
      Value dimSize = rewriter.create<DimOp>(loc, args[i], dim);
      Value lbCond =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::sge, dimArg, accBound);
      accBound = rewriter.create<AddIOp>(loc, accBound, dimSize);
      Value ubCond =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, dimArg, accBound);
      Value cond = rewriter.create<AndOp>(loc, lbCond, ubCond);
      // The first `nloops` arguments are indices.
      res = rewriter.create<SelectOp>(loc, cond, block->getArgument(nloops + i),
                                      res);
    }
    rewriter.create<linalg::YieldOp>(loc, res);
    rewriter.replaceOp(op, linalgOp.getResult(0));

    return success();
  }
};

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateHLOToLinalgOnTensorsConversionPatterns(&getContext(), patterns);

    ConversionTarget target(getContext());
    // Allow constant to appear in Linalg op regions.
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) -> bool {
      return isa<linalg::LinalgOp>(op.getOperation()->getParentOp());
    });
    // Don't convert the body of reduction ops.
    target.addDynamicallyLegalDialect<xla_hlo::XlaHloDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation* op) {
              auto parentOp = op->getParentRegion()->getParentOp();
              return isa<xla_hlo::ReduceOp>(parentOp) ||
                     isa<xla_hlo::ReduceWindowOp>(parentOp);
            }));
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext* context, OwningRewritePatternList& patterns) {
  xla_hlo::populateHLOToLinalgConversionPattern(context, &patterns);
  patterns.insert<SplatConstConverter, ConcatenateConverter>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>();
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors");

}  // namespace iree_compiler
}  // namespace mlir
