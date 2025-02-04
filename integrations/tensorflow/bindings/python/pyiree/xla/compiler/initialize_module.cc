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

#include <mutex>  // NOLINT

#include "bindings/python/pyiree/common/binding.h"
#include "bindings/python/pyiree/compiler/compiler.h"
#include "integrations/tensorflow/bindings/python/pyiree/xla/compiler/register_xla.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  m.doc() = "IREE XLA Compiler Interface";
  SetupCommonCompilerBindings(m);
  SetupXlaBindings(m);
}

}  // namespace python
}  // namespace iree
