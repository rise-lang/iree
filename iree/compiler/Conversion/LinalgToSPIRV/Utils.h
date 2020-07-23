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

//===- Utils.h - Utility functions used in Linalg to SPIR-V lowering ------===//
//
// Utility functions used while lowering from Linalg to SPIRV.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class FuncOp;
struct LogicalResult;

namespace iree_compiler {

/// Updates the workgroup size used for the dispatch region.
LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                  ArrayRef<int64_t> workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

#endif  //  IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_
