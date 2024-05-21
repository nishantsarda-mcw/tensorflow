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

#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool ContainsDimension(absl::Span<const Axis> dimensions, Axis dimension) {
  return std::find(dimensions.begin(), dimensions.end(), dimension) !=
         dimensions.end();
}

absl::InlinedVector<Axis, kMaxNumDimensions> CalculateResultDimensions(
    Axis rank, absl::Span<const Axis> batching_dimensions,
    absl::Span<const Axis> contracting_dimensions) {
  absl::InlinedVector<Axis, kMaxNumDimensions> result_dims;
  for (Axis i = 0; i < rank; ++i) {
    if (!ContainsDimension(batching_dimensions, i) &&
        !ContainsDimension(contracting_dimensions, i)) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

bool IncrementIndices(
    const Tensor& lhs,
    absl::InlinedVector<DimensionSize, kMaxNumDimensions>& lhs_index,
    absl::InlinedVector<DimensionSize, kMaxNumDimensions>& rhs_index,
    absl::Span<const Axis> lhs_contracting_dimensions,
    absl::Span<const Axis> rhs_contracting_dimensions, const size_t lhsc_size) {
  if (lhsc_size == 0) {
    return false;
  }
  for (int64_t i = static_cast<int64_t>(lhsc_size) - 1; i >= 0; --i) {
    lhs_index[lhs_contracting_dimensions[i]]++;
    rhs_index[rhs_contracting_dimensions[i]]++;
    if (lhs_index[lhs_contracting_dimensions[i]] <
        lhs.shape().Dim(lhs_contracting_dimensions[i])) {
      return true;
    }
    if (i == 0) {
      return false;
    }
    lhs_index[lhs_contracting_dimensions[i]] = 0;
    rhs_index[rhs_contracting_dimensions[i]] = 0;
  }
  return true;
}

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          absl::Span<const Axis> lhs_batching_dimensions,
                          absl::Span<const Axis> rhs_batching_dimensions,
                          absl::Span<const Axis> lhs_contracting_dimensions,
                          absl::Span<const Axis> rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  StorageT* output_data = output.GetDataAs<storage_type>();
  const DimensionSize output_size = output.NumElements();
  const Axis lhs_rank = lhs.Rank();
  const Axis rhs_rank = rhs.Rank();
  const Axis output_rank = output.Rank();
  const size_t lhsb_size = lhs_batching_dimensions.size();
  const size_t lhsc_size = lhs_contracting_dimensions.size();
  const size_t lhsr_size = op.lhs_result_dims.size();
  const size_t rhsr_size = op.rhs_result_dims.size();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_index(lhs_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_index(rhs_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
      output_rank);

  StorageT output_element(0);
  for (DimensionSize k = 0; k < output_size; ++k, ++output_data) {
    output.GetNdIndex(k, output_index);
    absl::c_fill(lhs_index, 0);
    absl::c_fill(rhs_index, 0);

    Axis result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      lhs_index[lhs_batching_dimensions[i]] = output_index[result_dim];
      rhs_index[rhs_batching_dimensions[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < lhsr_size; ++i, ++result_dim) {
      lhs_index[op.lhs_result_dims[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < rhsr_size; ++i, ++result_dim) {
      rhs_index[op.rhs_result_dims[i]] = output_index[result_dim];
    }
    output_element = 0;
    while (true) {
      output_element +=
          lhs.Get<storage_type>(lhs_index) * rhs.Get<storage_type>(rhs_index);
      if (!IncrementIndices(lhs, lhs_index, rhs_index,
                            lhs_contracting_dimensions,
                            rhs_contracting_dimensions, lhsc_size)) {
        break;
      }
    }
    *output_data = output_element;
  }
  return absl::OkStatus();
}

DotGeneralOp Create(DotGeneralOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  const Axis lhs_rank = lhs.Rank();
  const Axis rhs_rank = rhs.Rank();
  op.lhs_result_dims =
      CalculateResultDimensions(lhs_rank, op.attributes.lhs_batching_dimensions,
                                op.attributes.lhs_contracting_dimensions);
  op.rhs_result_dims =
      CalculateResultDimensions(rhs_rank, op.attributes.rhs_batching_dimensions,
                                op.attributes.rhs_contracting_dimensions);

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
