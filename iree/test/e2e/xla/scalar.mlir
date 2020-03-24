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

// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-use-linalg-to-spirv-path %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @scalar
func @scalar() -> i32 {
  %result = iree.unfoldable_constant 42 : i32
  return %result : i32
}
// CHECK: i32=42

