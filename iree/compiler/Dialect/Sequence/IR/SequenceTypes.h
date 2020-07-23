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

#ifndef IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCETYPES_H_
#define IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCETYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Sequence {
namespace detail {
struct SequenceTypeStorage;
}  // namespace detail

class SequenceType
    : public Type::TypeBase<SequenceType, Type, detail::SequenceTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Sequence; }
  static SequenceType get(Type targetType);
  static SequenceType getChecked(Type targetType, Location location);

  Type getTargetType();
};

}  // namespace Sequence
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCETYPES_H_
