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

#ifndef IREE_HAL_VMLA_VMLA_COMMAND_PROCESSOR_H_
#define IREE_HAL_VMLA_VMLA_COMMAND_PROCESSOR_H_

#include "iree/hal/host/serial_command_processor.h"
#include "iree/vm/stack.h"

namespace iree {
namespace hal {
namespace vmla {

class VMLACommandProcessor final : public SerialCommandProcessor {
 public:
  VMLACommandProcessor(Allocator* allocator,
                       CommandCategoryBitfield command_categories);
  ~VMLACommandProcessor() override;

  Status DispatchInline(
      Executable* executable, int32_t entry_point,
      std::array<uint32_t, 3> workgroups,
      const PushConstantBlock& push_constants,
      absl::Span<const absl::Span<const DescriptorSet::Binding>> set_bindings)
      override;

 private:
  iree_vm_stack_t* stack_ = nullptr;
};

}  // namespace vmla
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VMLA_VMLA_COMMAND_PROCESSOR_H_
