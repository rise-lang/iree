#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For use within a IREE bazel-swiftshader docker image on a Kokoro VM.
# Log some information about the environment, initialize the submodules and then
# run the bazel integrations tests.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
bazel --version
"$CXX" --version
"$CC" --version
"$PYTHON_BIN" -V
# TODO(#1875): Make PYTHON_BIN also control the runtime version
python3 -V

# Print Vulkan related information: SDK version and GPU ICD version
vulkaninfo 2>/dev/null | grep "Vulkan Instance" || echo "Vulkan Instance not found!"
vulkaninfo 2>/dev/null | grep -A7 "VkPhysicalDeviceProperties"  || echo "VkPhysicalDeviceProperties not found!"

# Print SwiftShader git commit
cat /swiftshader/git-commit

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

echo "Building and testing with bazel"
./build_tools/bazel/build_tensorflow.sh
