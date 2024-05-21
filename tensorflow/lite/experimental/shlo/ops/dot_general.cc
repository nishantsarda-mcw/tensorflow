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

bool HasInvalidDimension(absl::Span<const Axis> dimensions, const Axis rank) {
  return std::any_of(dimensions.begin(), dimensions.end(),
                     [=](Axis dim) { return dim < 0 || dim >= rank; });
}

bool ContainsDimension(absl::Span<const Axis> dimensions, Axis dimension) {
  return std::find(dimensions.begin(), dimensions.end(), dimension) !=
         dimensions.end();
}

bool HasUniqueDimension(absl::Span<const Axis> batching_dimensions,
                        absl::Span<const Axis> contracting_dimensions,
                        const size_t batch_size, const size_t contract_size) {
  std::unordered_set<Axis> batching_elements;
  std::unordered_set<Axis> contracting_elements;

  batching_elements.insert(batching_dimensions.begin(),
                           batching_dimensions.end());
  if (batching_dimensions.size() != batching_elements.size()) {
    return false;
  }
  contracting_elements.insert(contracting_dimensions.begin(),
                              contracting_dimensions.end());
  if (contracting_dimensions.size() != contracting_elements.size()) {
    return false;
  }
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < contract_size; ++j) {
      if (batching_dimensions[i] == contracting_dimensions[j]) {
        return false;
      }
    }
  }
  return true;
}

bool HasSameDimensionSize(const Tensor& lhs, const Tensor& rhs,
                          absl::Span<const Axis> lhs_dimensions,
                          absl::Span<const Axis> rhs_dimensions,
                          const size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (lhs.shape().Dim(lhs_dimensions[i]) !=
        rhs.shape().Dim(rhs_dimensions[i])) {
      return false;
    }
  }
  return true;
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

bool CheckZeroPoint(
    const QuantizedElementTypePerTensor::ZeroPointVariant& zero_point) {
  return std::visit(
      [](const auto& v) { return v == static_cast<decltype(v)>(0); },
      zero_point);
}

bool CheckZeroPoints(
    const QuantizedElementTypePerAxis::ZeroPointsVariant& zero_points) {
  return std::visit(
      [](const auto& v) {
        return std::all_of(v.begin(), v.end(), [](const auto value) {
          return value == static_cast<decltype(value)>(0);
        });
      },
      zero_points);
}

absl::Status CheckParameters(
    const Tensor& lhs, const Tensor& rhs,
    absl::Span<const Axis> lhs_batching_dimensions,
    absl::Span<const Axis> rhs_batching_dimensions,
    absl::Span<const Axis> lhs_contracting_dimensions,
    absl::Span<const Axis> rhs_contracting_dimensions, Tensor& output,
    const std::array<PrecisionTypes, 2>& precision_configs) {
  const size_t lhsb_size = lhs_batching_dimensions.size();
  const size_t rhsb_size = rhs_batching_dimensions.size();
  const size_t lhsc_size = lhs_contracting_dimensions.size();
  const size_t rhsc_size = rhs_contracting_dimensions.size();
  const Axis lhs_rank = lhs.Rank();
  const Axis rhs_rank = rhs.Rank();
  const Axis output_rank = output.Rank();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> expected_output_shape;

  if (precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of precision_config must be two.");
  }
  if (lhsb_size != rhsb_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_batching_dimensions and "
        "rhs_batching_dimensions must be same.");
  }
  if (lhsc_size != rhsc_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_contracting_dimensions and "
        "rhs_contracting_dimensions must be same.");
  }
  if (!HasUniqueDimension(lhs_batching_dimensions, lhs_contracting_dimensions,
                          lhsb_size, lhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs_batching_dimensions and "
        "lhs_contracting_dimensions must be unique.");
  }
  if (!HasUniqueDimension(rhs_batching_dimensions, rhs_contracting_dimensions,
                          rhsb_size, rhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The rhs_batching_dimensions and "
        "rhs_contracting_dimensions must be unique.");
  }
  if (HasInvalidDimension(lhs_batching_dimensions, lhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid lhs_batching_dimensions index.");
  }
  if (HasInvalidDimension(lhs_contracting_dimensions, lhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid lhs_contracting_dimensions index.");
  }
  if (HasInvalidDimension(rhs_batching_dimensions, rhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid rhs_batching_dimensions index.");
  }
  if (HasInvalidDimension(rhs_contracting_dimensions, rhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid rhs_contracting_dimensions index.");
  }
  if (!HasSameDimensionSize(lhs, rhs, lhs_batching_dimensions,
                            rhs_batching_dimensions, lhsb_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs and rhs tensors should have same batch "
        "dimension size.");
  }
  if (!HasSameDimensionSize(lhs, rhs, lhs_contracting_dimensions,
                            rhs_contracting_dimensions, lhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs and rhs tensors should have same "
        "contracting dimension size.");
  }

  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_result_dim =
      CalculateResultDimensions(lhs_rank, lhs_batching_dimensions,
                                lhs_contracting_dimensions);
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_result_dim =
      CalculateResultDimensions(rhs_rank, rhs_batching_dimensions,
                                rhs_contracting_dimensions);
  absl::Span<const Axis> lhs_result_dims(lhs_result_dim);
  absl::Span<const Axis> rhs_result_dims(rhs_result_dim);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_batch_dims_size =
      lhs.shape().Dims(lhs_batching_dimensions);
  expected_output_shape.insert(expected_output_shape.end(),
                               lhs_batch_dims_size.begin(),
                               lhs_batch_dims_size.end());
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_result_dims_size =
      lhs.shape().Dims(lhs_result_dims);
  expected_output_shape.insert(expected_output_shape.end(),
                               lhs_result_dims_size.begin(),
                               lhs_result_dims_size.end());
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_batch_dims_size =
      rhs.shape().Dims(rhs_result_dims);
  expected_output_shape.insert(expected_output_shape.end(),
                               rhs_batch_dims_size.begin(),
                               rhs_batch_dims_size.end());

  const Shape expected_output_check(expected_output_shape);
  if (expected_output_shape.size()) {
    if (expected_output_check != output.shape()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid output shape.");
    }
  }
  if (lhs.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs tensor cannot be per-axis quantized.");
  }
  if ((!lhs.IsQuantized() && !rhs.IsQuantized()) &&
      (lhs.tensor_element_type() != rhs.tensor_element_type())) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: For non-quantized tensors the element type of "
        "lhs and rhs must be the same.");
  }
  if (lhs.IsQuantized()) {
    if (!rhs.IsQuantized() || !output.IsQuantized()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If lhs and rhs are quantized tensors, than "
          "the output tensor should also be quantized.");
    }
    if (lhs.StorageType() != rhs.StorageType()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If the lhs and rhs are quantized tensors, "
          "than they should have the same storage type.");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!output.IsPerTensorQuantized()) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: If lhs and rhs are per-tensor quantized "
            "than output should also be per-tensor quantized.");
      }
      if ((lhs.ExpressedType() == rhs.ExpressedType()) &&
          (lhs.ExpressedType() != output.ExpressedType())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The expressed_type of output tensor must "
            "be the same as the expressed_type of lhs and rhs tensors.");
      }
      if (!CheckZeroPoint(
              rhs.quantized_per_tensor_element_type().ZeroPoint())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-tensor should have zero points "
            "as 0.");
      }
    } else if (rhs.IsPerAxisQuantized()) {
      if ((lhs.ExpressedType() == rhs.ExpressedType()) &&
          (lhs.ExpressedType() != output.ExpressedType())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The expressed_type of output must be the "
            "same as the expressed_type of lhs and rhs.");
      }
      if (!CheckZeroPoints(
              rhs.quantized_per_axis_element_type().ZeroPoints())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-axis should have zero points "
            "as 0.");
      }
      if (ContainsDimension(
              rhs_contracting_dimensions,
              rhs.quantized_per_axis_element_type().QuantizedDimension())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: If the rhs is per-axis quantized than the "
            "quantization_dimensions of rhs should not be in "
            "rhs_contracting_dimensions.");
      }
    }
  }
  return absl::OkStatus();
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
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();

  const DimensionSize batch_size = lhs.shape().Dim(0);
  const DimensionSize n = lhs.shape().Dim(1);
  const DimensionSize p = lhs.shape().Dim(2);
  const DimensionSize m = rhs.shape().Dim(1);
  const DimensionSize output_batch_size = n * m;
  const DimensionSize lhs_batch_size = n * p;
  const DimensionSize rhs_batch_size = m * p;

  memset(output_buffer, 0, sizeof(StorageT) * batch_size * output_batch_size);
  for (DimensionSize batch = 0; batch < batch_size; ++batch) {
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
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(lhs, rhs, op.attributes.lhs_batching_dimensions,
                      op.attributes.rhs_batching_dimensions,
                      op.attributes.lhs_contracting_dimensions,
                      op.attributes.rhs_contracting_dimensions, output,
                      op.attributes.precision_configs));
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
