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

#include "tensorflow/lite/experimental/shlo/ops/pad.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(const Tensor& operand, const Tensor& padding_value,
                             absl::Span<const DimensionSize> edge_padding_low,
                             absl::Span<const DimensionSize> edge_padding_high,
                             absl::Span<const DimensionSize> interior_padding,
                             const Tensor& output) {
  const size_t edge_padding_low_size = edge_padding_low.size();
  const size_t edge_padding_high_size = edge_padding_high.size();
  const size_t interior_padding_size = interior_padding.size();
  const Axis rank = operand.Rank();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> expected_output_shape(
      rank);

  if (operand.element_type() == padding_value.element_type() &&
      operand.element_type() != output.element_type()) {
    return absl::FailedPreconditionError(
        "stablehlo.pad: Element type of operand, padding_value and output "
        "tensors must be same.");
  }
  if (edge_padding_low_size == edge_padding_high_size &&
      edge_padding_low_size == interior_padding_size &&
      edge_padding_low_size != rank) {
    return absl::FailedPreconditionError(
        "stablehlo.pad: Size of edge_padding_low, edge_padding_high and "
        "interior_padding must be same as the rank of operand.");
  }
  for (size_t i = 0; i < interior_padding_size; ++i) {
    if (interior_padding[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.pad: Value of interior_padding must be more than 0.");
    }
  }
  for (Axis i = 0; i < rank; ++i) {
    expected_output_shape[i] =
        operand.shape().Dim(i) + edge_padding_low[i] +
        std::max<DimensionSize>(operand.shape().Dim(i) - 1, 0) *
            interior_padding[i] +
        edge_padding_high[i];
  }
  const Shape expected_output_check(expected_output_shape);
  if (expected_output_shape.size()) {
    if (expected_output_check != output.shape()) {
      return absl::FailedPreconditionError(
          "stablehlo.pad: Invalid output shape.");
    }
  }
  return absl::OkStatus();
}

DimensionSize DivNegRoundAwayOrZero(DimensionSize num, DimensionSize denum) {
  return num < 0 ? (num - denum + 1) / denum : 0;
}

template <typename StorageT>
void StridedCopy(const Axis rank, const StorageT* input,
                 const DimensionSize* modified_input_shape,
                 const DimensionSize* input_strides, StorageT* output,
                 const DimensionSize* output_strides,
                 const DimensionSize element_size, const DimensionSize depth) {
  if (depth + 1 == rank) {
    for (DimensionSize i = 0; i < modified_input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < modified_input_shape[depth]; ++i) {
      StridedCopy<StorageT>(rank, input, modified_input_shape, input_strides,
                            output, output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}

absl::Status PrepareImpl(PadOp& op, const Tensor& operand,
                         const Tensor& padding_value, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(CheckParameters(
      operand, padding_value, op.attributes.edge_padding_low,
      op.attributes.edge_padding_high, op.attributes.interior_padding, output));

  const Axis rank = operand.Rank();
  op.modified_input_shape.resize(rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_shape(rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_dimension_sizes(
      rank);
  // Compute the output shape.
  for (Axis i = 0; i < rank; ++i) {
    output_shape[i] =
        (operand.shape().Dim(i) - 1) * (op.attributes.interior_padding[i] + 1) +
        1 + op.attributes.edge_padding_low[i] +
        op.attributes.edge_padding_high[i];
  }
  if (absl::c_any_of(output_shape, [](DimensionSize s) { return s <= 0; })) {
    absl::c_fill(op.modified_input_shape, 0);
    absl::c_fill(output_shape, 0);
    op.output_size = 0;
    return;
  }
  output_dimension_sizes[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    output_dimension_sizes[i] =
        output_shape[i + 1] * output_dimension_sizes[i + 1];
  }
  // Compute the output stride for each dimension.
  // This is the stride between two elements that are copied from the input
  // tensor (i.e. not generated by interior padding).
  op.output_strides.resize(rank);
  op.output_strides[rank - 1] = op.attributes.interior_padding[rank - 1] + 1;
  for (int i = rank - 2; i >= 0; --i) {
    op.output_strides[i] =
        output_dimension_sizes[i] * (op.attributes.interior_padding[i] + 1);
  }
  // Compute the output offset from the eventual pads.
  for (Axis i = 0; i < rank; ++i) {
    op.output_offset +=
        std::max<DimensionSize>(op.attributes.edge_padding_low[i], 0) *
        output_dimension_sizes[i];
  }
  // Compute the final output size.
  op.output_size = absl::c_accumulate(output_shape, 1, std::multiplies<>());
  // Compute input strides.
  op.input_strides.resize(rank);
  op.input_strides[rank - 1] = 1;
  for (int i = rank - 1; i >= 1; --i) {
    op.input_strides[i - 1] = operand.shape().Dim(i) * op.input_strides[i];
  }
  // If negative padding is applied, the input shape is modified in place as
  // there is no use of it for anything else
  for (Axis i = 0; i < rank; ++i) {
    op.modified_input_shape[i] =
        operand.shape().Dim(i) +
        DivNegRoundAwayOrZero(op.attributes.edge_padding_low[i],
                              op.attributes.interior_padding[i] + 1) +
        DivNegRoundAwayOrZero(op.attributes.edge_padding_high[i],
                              op.attributes.interior_padding[i] + 1);
  }
  for (Axis i = 0; i < rank; ++i) {
    op.input_offset -=
        DivNegRoundAwayOrZero(op.attributes.edge_padding_low[i],
                              op.attributes.interior_padding[i] + 1) *
        op.input_strides[i];
    if (op.attributes.edge_padding_low[i] < 0) {
      DimensionSize tmp_offset = ((op.attributes.interior_padding[i] + 1 +
                                   op.attributes.edge_padding_low[i]) %
                                  (op.attributes.interior_padding[i] + 1));
      if (tmp_offset < 0) {
        tmp_offset += op.attributes.interior_padding[i] + 1;
      }
      op.output_offset += tmp_offset * output_dimension_sizes[i];
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(PadOp& op, const Tensor& operand,
                          const Tensor& padding_value,
                          absl::Span<const DimensionSize> edge_padding_low,
                          absl::Span<const DimensionSize> edge_padding_high,
                          absl::Span<const DimensionSize> interior_padding,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  const StorageT* padding_value_scalar =
      padding_value.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  // fill the output buffer with padding value
  std::fill(output_buffer, output_buffer + op.output_size,
            *padding_value_scalar);

  StridedCopy<StorageT>(operand.Rank(), operand_buffer + op.input_offset,
                        op.modified_input_shape.begin(),
                        op.input_strides.begin(),
                        output_buffer + op.output_offset,
                        op.output_strides.begin(), sizeof(StorageT),
                        /*depth=*/0);
  return absl::OkStatus();
}

PadOp Create(PadOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(PadOp& op, const Tensor& operand,
                     const Tensor& padding_value, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(PrepareImpl(op, operand, padding_value, output));
  return absl::OkStatus();
}

absl::Status Evaluate(PadOp& op, const Tensor& operand,
                      const Tensor& padding_value, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), op, operand,
                          padding_value, op.attributes.edge_padding_low,
                          op.attributes.edge_padding_high,
                          op.attributes.interior_padding, output);

  return absl::FailedPreconditionError(
      "stablehlo.pad: Unsupported tensor type.");
}

}  // namespace shlo_ref
