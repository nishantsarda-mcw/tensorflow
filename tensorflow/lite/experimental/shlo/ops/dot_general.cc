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
#include "ruy/ruy.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool HasInvalidDimension(const absl::Span<int64_t> dimensions,
                         const size_t rank) {
  return std::any_of(dimensions.begin(), dimensions.end(),
                     [=](int64_t dim) { return dim < 0 || dim >= rank; });
}

bool ContainsDimension(const absl::Span<int64_t> dimensions, size_t dimension) {
  return std::find(dimensions.begin(), dimensions.end(), dimension) !=
         dimensions.end();
}

bool HasUniqueDimension(const absl::Span<int64_t> batching_dimensions,
                        const absl::Span<int64_t> contracting_dimensions,
                        const DimensionSize batch_size,
                        const DimensionSize contract_size) {
  std::unordered_set<int64_t> batching_elements;
  std::unordered_set<int64_t> contracting_elements;

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
  for (DimensionSize i = 0; i < batch_size; ++i) {
    for (DimensionSize j = 0; j < contract_size; ++j) {
      if (batching_dimensions[i] == contracting_dimensions[j]) {
        return false;
      }
    }
  }
  return true;
}

bool HasSameDimensionSize(const Tensor& lhs, const Tensor& rhs,
                          const absl::Span<int64_t>& lhs_dimensions,
                          const absl::Span<int64_t>& rhs_dimensions,
                          const DimensionSize size) {
  for (DimensionSize i = 0; i < size; ++i) {
    if (lhs.shape().Dim(lhs_dimensions[i]) !=
        rhs.shape().Dim(rhs_dimensions[i])) {
      return false;
    }
  }
  return true;
}

absl::InlinedVector<size_t, 6> CalculateResultDimensions(
    size_t rank, absl::Span<int64_t> batching_dimensions,
    absl::Span<int64_t> contracting_dimensions) {
  absl::InlinedVector<size_t, 6> result_dims;
  for (size_t i = 0; i < rank; ++i) {
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

void GenerateIndices(size_t index, absl::InlinedVector<size_t, 6>& output_index,
                     const Tensor& output, const size_t output_rank) {
  size_t divisor = 1, dim = 0;
  for (size_t i = 0, j = output_rank - 1; i < output_rank; ++i, --j) {
    dim = output.shape().Dim(j);
    output_index[j] = (index / divisor) % dim;
    divisor *= dim;
  }
}

bool IncrementIndices(const Tensor& lhs,
                      absl::InlinedVector<size_t, 6>& lhs_index,
                      absl::InlinedVector<size_t, 6>& rhs_index,
                      const absl::Span<int64_t>& lhs_contracting_dimensions,
                      const absl::Span<int64_t>& rhs_contracting_dimensions,
                      const size_t lhsc_size) {
  if (lhsc_size == 0) return false;
  for (DimensionSize i = lhsc_size - 1; i >= 0; --i) {
    lhs_index[lhs_contracting_dimensions[i]]++;
    rhs_index[rhs_contracting_dimensions[i]]++;
    if (lhs_index[lhs_contracting_dimensions[i]] <
        lhs.shape().Dim(lhs_contracting_dimensions[i]))
      return true;
    if (i == 0) return false;
    lhs_index[lhs_contracting_dimensions[i]] = 0;
    rhs_index[rhs_contracting_dimensions[i]] = 0;
  }
  return true;
}

absl::Status CheckParameters(
    const Tensor& lhs, const Tensor& rhs,
    const absl::Span<int64_t> lhs_batching_dimensions,
    const absl::Span<int64_t> rhs_batching_dimensions,
    const absl::Span<int64_t> lhs_contracting_dimensions,
    const absl::Span<int64_t> rhs_contracting_dimensions, Tensor& output,
    const std::array<PrecisionTypes, 2>& precision_configs) {
  const DimensionSize lhsb_size = lhs_batching_dimensions.size();
  const DimensionSize rhsb_size = rhs_batching_dimensions.size();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.size();
  const DimensionSize rhsc_size = rhs_contracting_dimensions.size();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  absl::InlinedVector<size_t, 6> lhs_result_dims;
  absl::InlinedVector<size_t, 6> rhs_result_dims;
  std::vector<DimensionSize> expected_output_shape;

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

  lhs_result_dims = CalculateResultDimensions(lhs_rank, lhs_batching_dimensions,
                                              lhs_contracting_dimensions);
  rhs_result_dims = CalculateResultDimensions(rhs_rank, rhs_batching_dimensions,
                                              rhs_contracting_dimensions);
  for (size_t i = 0; i < lhsb_size; i++) {
    expected_output_shape.push_back(
        lhs.shape().Dim(lhs_batching_dimensions[i]));
  }
  for (size_t i = 0; i < lhs_result_dims.size(); ++i) {
    expected_output_shape.push_back(lhs.shape().Dim(lhs_result_dims[i]));
  }
  for (size_t i = 0; i < rhs_result_dims.size(); ++i) {
    expected_output_shape.push_back(rhs.shape().Dim(rhs_result_dims[i]));
  }
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
      if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
           rhs.quantized_per_tensor_element_type().ExpressedType()) &&
          (lhs.quantized_per_tensor_element_type().ExpressedType() !=
           output.quantized_per_tensor_element_type().ExpressedType())) {
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
      if (output.IsPerTensorQuantized()) {
        if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
             rhs.quantized_per_axis_element_type().ExpressedType()) &&
            (lhs.quantized_per_tensor_element_type().ExpressedType() !=
             output.quantized_per_tensor_element_type().ExpressedType())) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: The expressed_type of output must be the "
              "same as the expressed_type of lhs and rhs.");
        }
      } else if (output.IsPerAxisQuantized()) {
        if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
             rhs.quantized_per_axis_element_type().ExpressedType()) &&
            (lhs.quantized_per_tensor_element_type().ExpressedType() !=
             output.quantized_per_axis_element_type().ExpressedType())) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: The expressed_type of output must be the "
              "same as the expressed_type of lhs and rhs.");
        }
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
absl::Status PrepareTensors(DotGeneralOp& op, const Tensor& lhs,
                            const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const DimensionSize lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize lhsb_size = op.attributes.lhs_batching_dimensions.size();
  const DimensionSize rhsb_size = op.attributes.rhs_batching_dimensions.size();
  const DimensionSize lhsc_size =
      op.attributes.lhs_contracting_dimensions.size();
  const DimensionSize rhsc_size =
      op.attributes.rhs_contracting_dimensions.size();

  absl::InlinedVector<size_t, 6> newaxes_lhs;
  for (size_t i = 0; i < lhsb_size; ++i) {
    newaxes_lhs.push_back(op.attributes.lhs_batching_dimensions[i]);
  }
  for (size_t i = 0; i < op.lhs_result_dims.size(); ++i) {
    newaxes_lhs.push_back(op.lhs_result_dims[i]);
  }
  for (size_t i = 0; i < lhsc_size; ++i) {
    newaxes_lhs.push_back(op.attributes.lhs_contracting_dimensions[i]);
  }
  absl::InlinedVector<size_t, 6> newaxes_rhs;
  for (size_t i = 0; i < rhsb_size; ++i) {
    newaxes_rhs.push_back(op.attributes.rhs_batching_dimensions[i]);
  }
  for (size_t i = 0; i < op.rhs_result_dims.size(); ++i) {
    newaxes_rhs.push_back(op.rhs_result_dims[i]);
  }
  for (size_t i = 0; i < rhsc_size; ++i) {
    newaxes_rhs.push_back(op.attributes.rhs_contracting_dimensions[i]);
  }

  DimensionSize dim = 1;
  absl::InlinedVector<DimensionSize, 6> newshape_lhs;
  newshape_lhs.resize(3);
  if (lhsb_size == 0)
    newshape_lhs[0] = 1;
  else {
    for (size_t i = 0; i < lhsb_size; i++) {
      dim *= lhs.shape().Dim(op.attributes.lhs_batching_dimensions[i]);
    }
    newshape_lhs[0] = dim;
  }
  dim = 1;
  for (size_t i = 0; i < op.lhs_result_dims.size(); i++) {
    dim *= lhs.shape().Dim(op.lhs_result_dims[i]);
  }
  newshape_lhs[1] = dim;
  dim = 1;
  for (size_t i = 0; i < lhsc_size; i++) {
    dim *= lhs.shape().Dim(op.attributes.lhs_contracting_dimensions[i]);
  }
  newshape_lhs[2] = dim;

  absl::InlinedVector<DimensionSize, 6> newshape_rhs;
  newshape_rhs.resize(3);
  newshape_rhs[0] = newshape_lhs[0];
  dim = 1;
  for (size_t i = 0; i < op.rhs_result_dims.size(); i++) {
    dim *= rhs.shape().Dim(op.rhs_result_dims[i]);
  }
  newshape_rhs[1] = dim;
  dim = 1;
  for (size_t i = 0; i < rhsc_size; i++) {
    dim *= rhs.shape().Dim(op.attributes.rhs_contracting_dimensions[i]);
  }
  newshape_rhs[2] = dim;

  const Shape new_reshape_lhs(
      {newshape_lhs[0], newshape_lhs[1], newshape_lhs[2]});
  const Shape new_reshape_rhs(
      {newshape_rhs[0], newshape_rhs[1], newshape_rhs[2]});

  absl::InlinedVector<DimensionSize, 6> newaxes_lhs_shape;
  for (size_t i = 0; i < newaxes_lhs.size(); ++i) {
    newaxes_lhs_shape.push_back(lhs.shape().Dim(newaxes_lhs[i]));
  }
  absl::InlinedVector<DimensionSize, 6> newaxes_rhs_shape;
  for (size_t i = 0; i < newaxes_rhs.size(); ++i) {
    newaxes_rhs_shape.push_back(rhs.shape().Dim(newaxes_rhs[i]));
  }

  op.lhs_transposed_data = std::vector<std::byte>(lhs_size * sizeof(StorageT));
  const Shape lhs_transpose_shape(newaxes_lhs_shape);
  op.rhs_transposed_data = std::vector<std::byte>(rhs_size * sizeof(StorageT));
  const Shape rhs_transpose_shape(newaxes_rhs_shape);
  Tensor lhs_transpose{.type = TensorType{.shape = lhs_transpose_shape,
                                          .element_type = storage_type},
                       .data = op.lhs_transposed_data.data()};
  Tensor rhs_transpose{.type = TensorType{.shape = rhs_transpose_shape,
                                          .element_type = storage_type},
                       .data = op.rhs_transposed_data.data()};
  Tensor lhs_reshape{.type = TensorType{.shape = new_reshape_lhs,
                                        .element_type = storage_type},
                     .data = op.lhs_transposed_data.data()};
  Tensor rhs_reshape{.type = TensorType{.shape = new_reshape_rhs,
                                        .element_type = storage_type},
                     .data = op.rhs_transposed_data.data()};

  op.lhs_permutations = std::move(newaxes_lhs);
  op.rhs_permutations = std::move(newaxes_rhs);
  op.lhs_transposed = std::move(lhs_transpose);
  op.rhs_transposed = std::move(rhs_transpose);
  op.lhs_reshaped = std::move(lhs_reshape);
  op.rhs_reshaped = std::move(rhs_reshape);

  // quantized tensor prepare
  if (lhs.IsQuantized()) {
    op.lhs_dequantized_data =
        std::vector<std::byte>(lhs.NumElements() * sizeof(StorageT));
    const Shape lhs_dequantized_shape = lhs.shape();
    Tensor lhs_dequantized{.type = TensorType{.shape = lhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.lhs_dequantized_data.data()};
    op.rhs_dequantized_data =
        std::vector<std::byte>(rhs.NumElements() * sizeof(StorageT));
    const Shape rhs_dequantized_shape = rhs.shape();
    Tensor rhs_dequantized{.type = TensorType{.shape = rhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.rhs_dequantized_data.data()};
    op.output_dequantized_data =
        std::vector<std::byte>(output.NumElements() * sizeof(StorageT));
    const Shape output_dequantized_shape = output.shape();
    Tensor output_dequantized{
        .type = TensorType{.shape = output_dequantized_shape,
                           .element_type = storage_type},
        .data = op.output_dequantized_data.data()};

    op.lhs_dequantized = std::move(lhs_dequantized);
    op.rhs_dequantized = std::move(rhs_dequantized);
    op.output_dequantized = std::move(output_dequantized);
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status TransposeTensor(const Tensor& operand,
                             absl::InlinedVector<size_t, 6>& permutation,
                             Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();

  DimensionSize operand_element_index = 0, output_element_index = 0;
  DimensionSize operand_dim_accumulator = 1, output_dim_accumulator = 1;
  absl::InlinedVector<size_t, 6> operand_index_helper;
  operand_index_helper.resize(operand_rank);
  absl::InlinedVector<size_t, 6> output_index_helper;
  output_index_helper.resize(output_rank);
  absl::InlinedVector<size_t, 6> operand_index;
  absl::InlinedVector<size_t, 6> result_index;
  result_index.resize(output_rank);

  for (size_t i = 0; i < operand_rank; ++i) {
    operand_dim_accumulator *= operand.shape().Dim(i);
    operand_index_helper[i] = operand_size / operand_dim_accumulator;
    output_dim_accumulator *= output.shape().Dim(i);
    output_index_helper[i] = output_size / output_dim_accumulator;
  }

  for (size_t k = 0; k < operand_size; ++k) {
    GenerateIndices(k, operand_index, operand, operand_rank);
    absl::c_fill(result_index, 0);

    for (size_t d = 0; d < output_rank; ++d) {
      result_index[d] = operand_index[permutation[d]];
    }
    operand_element_index = 0;
    output_element_index = 0;
    for (size_t i = 0; i < operand_rank; ++i) {
      operand_element_index += operand_index[i] * operand_index_helper[i];
      output_element_index += result_index[i] * output_index_helper[i];
    }
    output_buffer[output_element_index] = operand_buffer[operand_element_index];
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status ReshapeTensor(const Tensor& operand, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();

  DimensionSize operand_element_index = 0, output_element_index = 0;
  DimensionSize operand_dim_accumulator = 1, output_dim_accumulator = 1;
  absl::InlinedVector<size_t, 6> operand_index;
  absl::InlinedVector<size_t, 6> output_index;
  absl::InlinedVector<size_t, 6> operand_index_helper;
  operand_index_helper.resize(operand_rank);
  absl::InlinedVector<size_t, 6> output_index_helper;
  output_index_helper.resize(output_rank);

  for (size_t i = 0; i < operand_rank; ++i) {
    operand_dim_accumulator *= operand.shape().Dim(i);
    operand_index_helper[i] = operand_size / operand_dim_accumulator;
  }
  for (size_t i = 0; i < output_rank; ++i) {
    output_dim_accumulator *= output.shape().Dim(i);
    output_index_helper[i] = output_size / output_dim_accumulator;
  }

  for (size_t k = 0; k < operand_size; ++k) {
    GenerateIndices(k, operand_index, operand, operand_rank);
    GenerateIndices(k, output_index, output, output_rank);

    operand_element_index = 0;
    output_element_index = 0;
    for (size_t i = 0; i < operand_rank; ++i) {
      operand_element_index += operand_index[i] * operand_index_helper[i];
    }
    for (size_t i = 0; i < output_rank; ++i) {
      output_element_index += output_index[i] * output_index_helper[i];
    }
    output_buffer[output_element_index] = operand_buffer[operand_element_index];
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          const absl::Span<int64_t> lhs_batching_dimensions,
                          const absl::Span<int64_t> rhs_batching_dimensions,
                          const absl::Span<int64_t> lhs_contracting_dimensions,
                          const absl::Span<int64_t> rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const DimensionSize lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  const DimensionSize lhsb_size = lhs_batching_dimensions.size();
  const DimensionSize rhsb_size = rhs_batching_dimensions.size();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.size();
  const DimensionSize rhsc_size = rhs_contracting_dimensions.size();
  // should i have diff status variable for each case?
  absl::Status status;
  status = TransposeTensor<storage_type>(lhs, op.lhs_permutations,
                                         op.lhs_transposed);
  status = ReshapeTensor<storage_type>(op.lhs_transposed, op.lhs_reshaped);
  status = TransposeTensor<storage_type>(rhs, op.rhs_permutations,
                                         op.rhs_transposed);
  status = ReshapeTensor<storage_type>(op.rhs_transposed, op.rhs_reshaped);

  StorageT* lhs_reshape_buffer = op.lhs_reshaped.GetDataAs<storage_type>();
  StorageT* rhs_reshape_buffer = op.rhs_reshaped.GetDataAs<storage_type>();

  const size_t batchsize = op.lhs_reshaped.shape().Dim(0);
  const size_t n = op.lhs_reshaped.shape().Dim(1);
  const size_t m = op.rhs_reshaped.shape().Dim(1);
  const size_t p = op.lhs_reshaped.shape().Dim(2);
  const size_t size = n * m;

  //                              BRUTE FORCE IMPLEMENTATION

  // for (size_t batch = 0; batch < batchsize; ++batch) {
  //   for (size_t i = 0; i < n; ++i) {
  //     for (size_t j = 0; j < m; ++j) {
  //       output_data[batch * size + i * m + j] = 0;
  //       for (size_t k = 0; k < p; ++k) {
  //         output_data[batch * size + i * m + j] +=
  //             lhs_reshape_buffer[batch * n * p + i * p + k] *
  //             rhs_reshape_buffer[batch * m * p + j * p + k];
  //       }
  //     }
  //   }
  // }

  //                              TILED IMPLEMENTATION

  // const size_t kBlockSize = 32;// we can change
  // for (size_t batch = 0; batch < batchsize; ++batch) {
  //   for (size_t ii = 0; ii < n; ii += kBlockSize) {
  //     for (size_t jj = 0; jj < m; jj += kBlockSize) {
  //       for (size_t kk = 0; kk < p; kk += kBlockSize) {
  //         const size_t i_block = std::min(ii + kBlockSize, n);
  //         const size_t j_block = std::min(jj + kBlockSize, m);
  //         const size_t k_block = std::min(kk + kBlockSize, p);
  //         for (size_t i = ii; i < i_block; ++i) {
  //           for (size_t j = jj; j < j_block; ++j) {
  //             for (size_t k = kk; k < k_block; ++k) {
  //               output_data[batch * size + i * m + j] +=
  //                   lhs_reshape_buffer[batch * n * p + i * p + k] *
  //                   rhs_reshape_buffer[batch * m * p + j * p + k];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  //                           EIGEN IMPLEMENTATION

  //                           RUY IMPLEMENTATION
  // const size_t batchsize = lhs.shape().Dim(0);  // batch dim
  // const size_t n = lhs.shape().Dim(1);          // lhs result dim
  // const size_t m = rhs.shape().Dim(1);          // rhs result dim
  // const size_t p = lhs.shape().Dim(2);          // contracting dim

  // for (int batch = 0; batch < batchsize; ++batch) {
  //   ruy::Matrix<float> lhs_matrix;
  //   ruy::MakeSimpleLayout(n, p, ruy::Order::kRowMajor,
  //                         lhs_matrix.mutable_layout());
  //   lhs_matrix.set_data(lhs_data + batch * n * p);
  //   ruy::Matrix<float> rhs_matrix;
  //   ruy::MakeSimpleLayout(p, m, ruy::Order::kRowMajor,
  //                         rhs_matrix.mutable_layout());
  //   rhs_matrix.set_data(rhs_data + batch * m * p);
  //   ruy::Matrix<float> output_matrix;
  //   ruy::MakeSimpleLayout(n, m, ruy::Order::kColMajor,
  //                         output_matrix.mutable_layout());
  //   output_matrix.set_data(output_data + batch * n * m);
  //   ruy::MulParams<float, float> mul_params;
  //   ruy::Context context;
  //   ruy::Mul(lhs_matrix, rhs_matrix, mul_params, &context, &output_matrix);
  // }

  //                          REFERENCE IMPLEMENTATION

  absl::InlinedVector<size_t, 6> lhs_index;
  lhs_index.resize(lhs_rank);
  absl::InlinedVector<size_t, 6> rhs_index;
  rhs_index.resize(rhs_rank);
  absl::InlinedVector<size_t, 6> lhs_index_helper;
  lhs_index_helper.resize(lhs_rank);
  absl::InlinedVector<size_t, 6> rhs_index_helper;
  rhs_index_helper.resize(rhs_rank);
  absl::InlinedVector<size_t, 6> output_index;
  output_index.resize(output_rank);
  absl::c_fill(output_index, 0);

  DimensionSize lhs_dim_accumulator = 1, rhs_dim_accumulator = 1;
  for (size_t i = 0; i < lhs_rank; ++i) {
    lhs_dim_accumulator *= lhs.shape().Dim(i);
    lhs_index_helper[i] = lhs_size / lhs_dim_accumulator;
  }
  for (size_t i = 0; i < rhs_rank; ++i) {
    rhs_dim_accumulator *= rhs.shape().Dim(i);
    rhs_index_helper[i] = rhs_size / rhs_dim_accumulator;
  }

  StorageT output_element(0);
  DimensionSize lhs_element_index = 0, rhs_element_index = 0;
  for (DimensionSize k = 0; k < output_size; ++k, ++output_data) {
    GenerateIndices(k, output_index, output, output_rank);
    absl::c_fill(lhs_index, 0);
    absl::c_fill(rhs_index, 0);

    size_t result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      lhs_index[lhs_batching_dimensions[i]] = output_index[result_dim];
      rhs_index[rhs_batching_dimensions[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.lhs_result_dims.size(); ++i, ++result_dim) {
      lhs_index[op.lhs_result_dims[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.rhs_result_dims.size(); ++i, ++result_dim) {
      rhs_index[op.rhs_result_dims[i]] = output_index[result_dim];
    }
    output_element = 0;
    while (true) {
      lhs_element_index = 0;
      rhs_element_index = 0;
      for (size_t i = 0; i < lhs_rank; ++i) {
        lhs_element_index += lhs_index[i] * lhs_index_helper[i];
      }
      for (size_t i = 0; i < rhs_rank; ++i) {
        rhs_element_index += rhs_index[i] * rhs_index_helper[i];
      }
      output_element +=
          lhs_data[lhs_element_index] * rhs_data[rhs_element_index];

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

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerTensor(DotGeneralOp& op, const Tensor& lhs,
                                   const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      op.lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      op.rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data) {
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
  }

  const DimensionSize rhs_num_elements = rhs.NumElements();
  const StorageT rhs_zero_point =
      rhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT rhs_scale =
      rhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < rhs_num_elements;
       ++i, ++rhs_data, ++rhs_dequantized_data) {
    *rhs_dequantized_data = Dequantize(*rhs_data, rhs_zero_point, rhs_scale);
  }

  absl::Status status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
                                 op.output_dequantized);

  const DimensionSize output_num_elements = output.NumElements();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

  for (DimensionSize i = 0; i < output_num_elements;
       ++i, ++output_dequantized_data, ++output_data) {
    *output_data = Quantize<storage_type, expressed_type>(
        *output_dequantized_data, output_zero_point, inv_scale);
  }
}

template <typename StorageT, typename ExpressedT>
void DequantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    const StorageT* input_data, ExpressedT* inputDeQuantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *inputDeQuantized_data =
          Dequantize(*input_data, input_zero_points[quantization_index],
                     input_scales[quantization_index]);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      DequantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDeQuantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  }
}

template <typename StorageT, typename ExpressedT>
void QuantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    StorageT* input_data, const ExpressedT* inputDequantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *input_data = Quantize<StorageT, ExpressedT>(
          *inputDequantized_data, input_zero_points[quantization_index],
          static_cast<ExpressedT>(1 / input_scales[quantization_index]),
          quantization_min, quantization_max);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      QuantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDequantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  }
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerAxis(DotGeneralOp& op, const Tensor& lhs,
                                 const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      op.lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      op.rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data) {
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
  }

  const Shape& shape = rhs.shape();
  const Axis rhs_quantization_dimension =
      rhs.quantized_per_axis_element_type().QuantizedDimension();
  const absl::Span<const StorageT> rhs_zero_points =
      rhs.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> rhs_scales =
      rhs.quantized_per_axis_element_type().ScalesAs<expressed_type>();
  const Strides& strides = ComputeStrides(shape);
  DequantizeOpQuantizePerAxisImpl(
      shape, rhs_quantization_dimension, Storage<storage_type>::kMinValue,
      Storage<storage_type>::kMaxValue, rhs_zero_points, rhs_scales, strides,
      rhs_data, rhs_dequantized_data, /*depth=*/0, /*quantization_index=*/0);

  absl::Status status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
                                 op.output_dequantized);
  if (output.IsPerAxisQuantized()) {
    const Shape& shape = output.shape();
    const Axis output_quantization_dimension =
        output.quantized_per_axis_element_type().QuantizedDimension();
    const absl::Span<const StorageT> output_zero_points =
        output.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
    const absl::Span<const ExpressedT> output_scales =
        output.quantized_per_axis_element_type().ScalesAs<expressed_type>();
    const Strides& strides = ComputeStrides(shape);
    QuantizeOpQuantizePerAxisImpl(
        shape, output_quantization_dimension, Storage<storage_type>::kMinValue,
        Storage<storage_type>::kMaxValue, output_zero_points, output_scales,
        strides, output_data, output_dequantized_data, /*depth=*/0,
        /*quantization_index=*/0);
  } else {
    const DimensionSize output_num_elements = output.NumElements();
    const StorageT output_zero_point =
        output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
    const ExpressedT output_scale =
        output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
    const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

    for (DimensionSize i = 0; i < output_num_elements;
         ++i, ++output_dequantized_data, ++output_data) {
      *output_data = Quantize<storage_type, expressed_type>(
          *output_dequantized_data, output_zero_point, inv_scale);
    }
  }
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

  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  op.lhs_result_dims =
      CalculateResultDimensions(lhs_rank, op.attributes.lhs_batching_dimensions,
                                op.attributes.lhs_contracting_dimensions);
  op.rhs_result_dims =
      CalculateResultDimensions(rhs_rank, op.attributes.rhs_batching_dimensions,
                                op.attributes.rhs_contracting_dimensions);

  if (lhs.IsQuantized()) {
    DISPATCH_BOOL_INT_FLOAT(
        PrepareTensors, lhs.quantized_per_tensor_element_type().ExpressedType(),
        op, lhs, rhs, output);
  }
  if (!lhs.IsQuantized()) {
    DISPATCH_BOOL_INT_FLOAT(PrepareTensors, lhs.StorageType(), op, lhs, rhs,
                            output);
  }
  return absl::OkStatus();
}

absl::Status Evaluate(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (lhs.IsQuantized()) {
    if (rhs.IsPerTensorQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerTensor,
          lhs.quantized_per_tensor_element_type().StorageType(),
          lhs.quantized_per_tensor_element_type().ExpressedType(), op, lhs, rhs,
          output);
    } else if (rhs.IsPerAxisQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerAxis,
          lhs.quantized_per_tensor_element_type().StorageType(),
          lhs.quantized_per_tensor_element_type().ExpressedType(), op, lhs, rhs,
          output);
    }
  } else {
    DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs,
                            rhs, op.attributes.lhs_batching_dimensions,
                            op.attributes.rhs_batching_dimensions,
                            op.attributes.lhs_contracting_dimensions,
                            op.attributes.rhs_contracting_dimensions, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}

}  // namespace shlo_ref
