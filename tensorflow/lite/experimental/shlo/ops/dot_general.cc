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

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          absl::Span<const Axis> lhs_batching_dimensions,
                          absl::Span<const Axis> rhs_batching_dimensions,
                          absl::Span<const Axis> lhs_contracting_dimensions,
                          absl::Span<const Axis> rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();

  const DimensionSize batchsize = lhs.shape().Dim(0);
  const DimensionSize n = lhs.shape().Dim(1);
  const DimensionSize p = lhs.shape().Dim(2);
  const DimensionSize m = rhs.shape().Dim(1);
  const DimensionSize output_batch_size = n * m;
  const DimensionSize lhs_batch_size = n * p;
  const DimensionSize rhs_batch_size = m * p;

  memset(output_buffer,0,sizeof(StorageT)*output.NumElements());
  for (DimensionSize batch = 0; batch < batchsize; ++batch) {
    for (DimensionSize i = 0; i < n; ++i) {
      for (DimensionSize k = 0; k < p; ++k) {
        for (DimensionSize j = 0; j < m; ++j) {
          output_buffer[batch * output_batch_size + i * m + j] +=
              lhs_buffer[batch * lhs_batch_size + i * p + k] *
              rhs_buffer[batch * rhs_batch_size + j * p + k];
        }
      }
    }
  }
  return absl::OkStatus();
}

DotGeneralOp Create(DotGeneralOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  return absl::OkStatus();
}

absl::Status Evaluate(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
    DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs,
                            rhs, op.attributes.lhs_batching_dimensions,
                            op.attributes.rhs_batching_dimensions,
                            op.attributes.lhs_contracting_dimensions,
                            op.attributes.rhs_contracting_dimensions, output);
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}

}  // namespace shlo_ref
