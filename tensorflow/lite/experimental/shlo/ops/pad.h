/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_PAD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_PAD_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

class PadOp {
 public:
  struct Attributes {
    absl::Span<const DimensionSize> edge_padding_low;
    absl::Span<const DimensionSize> edge_padding_high;
    absl::Span<const DimensionSize> interior_padding;
  };
  Attributes attributes;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> modified_input_shape;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> input_strides;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_strides;
  DimensionSize input_offset;
  DimensionSize output_offset;
  DimensionSize output_size;
};

PadOp Create(PadOp::Attributes attributes);
absl::Status Prepare(PadOp& op, const Tensor& operand,
                     const Tensor& padding_value, Tensor& output);
absl::Status Evaluate(PadOp& op, const Tensor& operand,
                      const Tensor& padding_value, Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_PAD_H_
