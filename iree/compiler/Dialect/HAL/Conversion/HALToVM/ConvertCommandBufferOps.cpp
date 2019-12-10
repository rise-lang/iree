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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class RemoveMakeMemoryBarrierOpConversion
    : public OpConversionPattern<IREE::HAL::MakeMemoryBarrierOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      IREE::HAL::MakeMemoryBarrierOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

class CommandBufferExecutionBarrierOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferExecutionBarrierOp> {
 public:
  CommandBufferExecutionBarrierOpConversion(MLIRContext *context,
                                            SymbolTable &importSymbols,
                                            TypeConverter &typeConverter,
                                            StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  PatternMatchResult matchAndRewrite(
      IREE::HAL::CommandBufferExecutionBarrierOp op,
      llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();

    SmallVector<Value *, 8> callOperands = {
        operands[0],
        rewriter.create<mlir::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(op.source_stage_mask()))),
        rewriter.create<mlir::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(op.target_stage_mask()))),
    };
    SmallVector<int8_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*source_stage_mask=*/-1,
        /*target_stage_mask=*/-1,
        /*memory_barriers=*/
        static_cast<int8_t>(std::distance(op.memory_barriers().begin(),
                                          op.memory_barriers().end())),
        /*buffer_barriers=*/
        static_cast<int8_t>(std::distance(op.buffer_barriers().begin(),
                                          op.buffer_barriers().end())),
    };
    if (!op.buffer_barriers().empty()) {
      op.emitOpError()
          << "tuples not yet fully supported; don't use buffer barriers";
      return matchFailure();
    }
    for (auto *memoryBarrier : op.memory_barriers()) {
      assert(memoryBarrier->getDefiningOp());
      auto makeMemoryBarrierOp =
          cast<IREE::HAL::MakeMemoryBarrierOp>(memoryBarrier->getDefiningOp());
      callOperands.push_back(rewriter.create<mlir::ConstantOp>(
          op.getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                           makeMemoryBarrierOp.source_scope()))));
      callOperands.push_back(rewriter.create<mlir::ConstantOp>(
          op.getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                           makeMemoryBarrierOp.target_scope()))));
    }

    rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
    return matchSuccess();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALCommandBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<RemoveMakeMemoryBarrierOpConversion>(context);

  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCreateOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferBeginOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.begin");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferEndOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.end");
  patterns.insert<CommandBufferExecutionBarrierOpConversion>(
      context, importSymbols, typeConverter,
      "_hal.command_buffer.execution_barrier");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferFillBufferOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.fill_buffer");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCopyBufferOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.copy_buffer");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::CommandBufferBindDescriptorSetOp>>(
      context, importSymbols, typeConverter,
      "_hal.command_buffer.bind_descriptor_set");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchOp>>(
      context, importSymbols, typeConverter, "_hal.command_buffer.dispatch");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchIndirectOp>>(
          context, importSymbols, typeConverter,
          "_hal.command_buffer.dispatch.indirect");
}

}  // namespace iree_compiler
}  // namespace mlir