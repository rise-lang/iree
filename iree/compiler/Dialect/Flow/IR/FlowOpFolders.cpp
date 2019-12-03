// Copyright 2019 Google LLC
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

#include <algorithm>
#include <numeric>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

namespace {

/// Converts variable initializer functions that evaluate to a constant to a
/// specified initial value.
struct InlineConstVariableOpInitializer : public OpRewritePattern<VariableOp> {
  using OpRewritePattern<VariableOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(VariableOp op,
                                     PatternRewriter &rewriter) const override {
    if (!op.initializer()) return matchFailure();
    auto *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue());
    auto initializer = cast<FuncOp>(symbolOp);
    if (initializer.getBlocks().size() == 1 &&
        initializer.getBlocks().front().getOperations().size() == 2 &&
        isa<mlir::ReturnOp>(
            initializer.getBlocks().front().getOperations().back())) {
      auto &primaryOp = initializer.getBlocks().front().getOperations().front();
      Attribute constResult;
      if (matchPattern(primaryOp.getResult(0), m_Constant(&constResult))) {
        rewriter.replaceOpWithNewOp<VariableOp>(
            op, op.sym_name(), op.is_mutable(), op.type(), constResult);
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void VariableOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<InlineConstVariableOpInitializer>(context);
}

namespace {

/// Erases flow.variable.load ops whose values are unused.
/// We have to do this manually as the load op cannot be marked pure and have it
/// done automatically.
struct EraseUnusedVariableLoadOp : public OpRewritePattern<VariableLoadOp> {
  using OpRewritePattern<VariableLoadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(VariableLoadOp op,
                                     PatternRewriter &rewriter) const override {
    if (op.result()->use_empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }
    return matchFailure();
  }
};

}  // namespace

void VariableLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableLoadOp>(context);
}

namespace {

/// Erases flow.variable.store ops that are no-ops.
/// This can happen if there was a variable load, some DCE'd usage, and a
/// store back to the same variable: we want to be able to elide the entire load
/// and store.
struct EraseUnusedVariableStoreOp : public OpRewritePattern<VariableStoreOp> {
  using OpRewritePattern<VariableStoreOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(VariableStoreOp op,
                                     PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<VariableLoadOp>(op.value()->getDefiningOp())) {
      if (loadOp.variable() == op.variable()) {
        rewriter.eraseOp(op);
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void VariableStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableStoreOp>(context);
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

/// Reduces the provided multidimensional index into a flattended 1D row-major
/// index. The |type| is expected to be statically shaped (as all constants
/// are).
static uint64_t getFlattenedIndex(ShapedType type, ArrayRef<uint64_t> index) {
  assert(type.hasStaticShape() && "for use on statically shaped types only");
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto sourceType = source()->getType().cast<ShapedType>();
  auto resultType = result()->getType().cast<ShapedType>();
  if (sourceType.hasStaticShape() && sourceType == resultType) {
    // No-op.
    return source();
  }

  // Skip intermediate reshapes.
  if (auto definingOp =
          dyn_cast_or_null<TensorReshapeOp>(source()->getDefiningOp())) {
    setOperand(definingOp.getOperand());
    return result();
  }

  return {};
}

OpFoldResult TensorLoadOp::fold(ArrayRef<Attribute> operands) {
  if (auto source = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    // Load directly from the constant source tensor.
    auto indices = operands.drop_front();
    if (llvm::count(indices, nullptr) == 0) {
      return source.getValue(
          llvm::to_vector<4>(llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          })));
    }
  }
  return {};
}

OpFoldResult TensorStoreOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};
  auto &value = operands[0];
  if (auto target = operands[1].dyn_cast_or_null<ElementsAttr>()) {
    // Store into the constant target tensor.
    if (target.getType().getRank() == 0) {
      return DenseElementsAttr::get(target.getType(), {value});
    }
    auto indices = operands.drop_front(2);
    if (llvm::count(indices, nullptr) == 0) {
      uint64_t offset = getFlattenedIndex(
          target.getType(),
          llvm::to_vector<4>(llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          })));
      SmallVector<Attribute, 16> newContents(target.getValues<Attribute>());
      newContents[offset] = value;
      return DenseElementsAttr::get(target.getType(), newContents);
    }
  }
  return {};
}

OpFoldResult TensorSplatOp::fold(ArrayRef<Attribute> operands) {
  // TODO(benvanik): only fold when shape is constant.
  if (operands[0]) {
    // Splat value is constant and we can fold the operation.
    return SplatElementsAttr::get(result()->getType().cast<ShapedType>(),
                                  operands[0]);
  }
  return {};
}

OpFoldResult TensorCloneOp::fold(ArrayRef<Attribute> operands) {
  if (operands[0]) {
    return operands[0];
  }
  // TODO(benvanik): fold if clone device placements differ.
  return operand();
}

OpFoldResult TensorSliceOp::fold(ArrayRef<Attribute> operands) {
  if (operands[0] && operands[1] && operands[2]) {
    // Fully constant arguments so we can perform the slice here.
    // TODO(benvanik): constant slice.
    return {};
  }
  return {};
}

static ElementsAttr tensorUpdate(ElementsAttr update, ElementsAttr target,
                                 ArrayRef<Attribute> startIndicesAttrs) {
  // TODO(benvanik): tensor update constant folding.
  return {};
}

OpFoldResult TensorUpdateOp::fold(ArrayRef<Attribute> operands) {
  auto indices = operands.drop_front(2);
  bool allIndicesConstant = llvm::count(indices, nullptr) == 0;
  if (operands[0] && operands[1] && allIndicesConstant) {
    // Fully constant arguments so we can perform the update here.
    return tensorUpdate(operands[0].cast<ElementsAttr>(),
                        operands[1].cast<ElementsAttr>(), indices);
  } else {
    // Replace the entire tensor when the sizes match.
    auto updateType = update()->getType().cast<ShapedType>();
    auto targetType = target()->getType().cast<ShapedType>();
    if (updateType.hasStaticShape() && targetType.hasStaticShape() &&
        updateType == targetType) {
      return update();
    }
  }
  return {};
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir