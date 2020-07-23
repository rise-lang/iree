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
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_AOT_TARGET_LINKER_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_AOT_TARGET_LINKER_H_

#include <string>

#include "iree/base/file_io.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Calls linker tool to link objData and returns shared library blob.
iree::StatusOr<std::string> linkLLVMAOTObjects(
    const std::string& linkerToolPath, const std::string& objData);
// Use lld::elf::link for linking objData and returns shared library blob.
iree::StatusOr<std::string> linkLLVMAOTObjectsWithLLDElf(
    const std::string& objData);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_AOT_TARGET_LINKER_H_
