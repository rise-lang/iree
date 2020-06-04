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

#ifndef IREE_HAL_DYLIB_EXECUTABLE_CACHE_H_
#define IREE_HAL_DYLIB_EXECUTABLE_CACHE_H_

#include "iree/hal/executable.h"
#include "iree/hal/executable_cache.h"

namespace iree {
namespace hal {
namespace dylib {

class DyLibExecutableCache final : public ExecutableCache {
 public:
  DyLibExecutableCache();
  ~DyLibExecutableCache() override;

  bool CanPrepareFormat(ExecutableFormat format) const override;

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
      const ExecutableSpec& spec) override;
};

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_EXECUTABLE_CACHE_H_
