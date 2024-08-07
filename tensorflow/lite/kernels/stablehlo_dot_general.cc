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

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_dot_general {
namespace {

static constexpr int kMaxDims = 6;

using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;

static bool IsQuantized(const TfLiteTensor* input) {
  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto* quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 0);
  }
  return false;
}

static bool HasInvalidDimension(const int64_t* dimensions, const int size,
                                const int rank) {
  return std::any_of(dimensions, dimensions + size,
                     [=](const int64_t dim) { return dim < 0 || dim > rank; });
}

static bool HasSameDimensionSize(const TfLiteTensor* lhs,
                                 const TfLiteTensor* rhs,
                                 const int64_t* lhs_dimensions,
                                 const int64_t* rhs_dimensions,
                                 const int dimension_array_size) {
  for (int i = 0; i < dimension_array_size; ++i) {
    if (lhs->dims->data[lhs_dimensions[i]] !=
        rhs->dims->data[rhs_dimensions[i]]) {
      return false;
    }
  }
  return true;
}

static bool HasUniqueDimensions(const int64_t* batching_dimensions,
                                const int batching_array_size,
                                const int64_t* contracting_dimensions,
                                const int contracting_array_size) {
  std::unordered_set<int64_t> batching_dimensions_set;
  std::unordered_set<int64_t> contracting_dimensions_set;
  batching_dimensions_set.insert(batching_dimensions,
                                 batching_dimensions + batching_array_size);
  if (batching_dimensions_set.size() != batching_array_size) {
    return false;
  }
  contracting_dimensions_set.insert(
      contracting_dimensions, contracting_dimensions + contracting_array_size);
  if (contracting_dimensions_set.size() != contracting_array_size) {
    return false;
  }
  for (int i = 0; i < batching_array_size; ++i) {
    for (int j = 0; j < contracting_array_size; ++j) {
      if (batching_dimensions[i] == contracting_dimensions[j]) {
        return false;
      }
    }
  }
  return true;
}

static std::vector<int64_t> CalculateResultDimensions(
    const int rank, const int64_t* batching_dimensions,
    const int batching_array_size, const int64_t* contracting_dimensions,
    const int contracting_array_size) {
  std::vector<int64_t> result_dims;
  for (int64_t i = 0; i < rank; ++i) {
    if (!ArrayContains(batching_dimensions, batching_array_size, i) &&
        !ArrayContains(contracting_dimensions, contracting_array_size, i)) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

struct DotGeneralData {
 public:
  enum { kLhsTensor, kRhsTensor };
  enum { kOutputTensor };
  // The index of the temporary tensors to store transposed LHS/RHS.
  int scratch_tensor_index;

  TfLiteStatus CheckParameters(TfLiteContext* context,
                               const TfLiteStablehloDotGeneralParams* params,
                               const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                               TfLiteTensor* output) {
    lhs_rank_ = lhs->dims->size;
    rhs_rank_ = rhs->dims->size;
    lhsb_size_ = params->num_lhs_batching_dimensions;
    rhsb_size_ = params->num_rhs_batching_dimensions;
    lhsc_size_ = params->num_lhs_contracting_dimensions;
    rhsc_size_ = params->num_rhs_contracting_dimensions;

    TF_LITE_ENSURE_MSG(context, params->num_precision_configs == 2,
                       "'stablehlo.dot_general' size of precision_config "
                       "parameter must be two.");
    TF_LITE_ENSURE_MSG(
        context, lhsb_size_ == rhsb_size_,
        "'stablehlo.dot_general' size of lhs_batching_dimensions and "
        "rhs_batching_dimensions must be the same.");
    TF_LITE_ENSURE_MSG(
        context, lhsc_size_ == rhsc_size_,
        "'stablehlo.dot_general' size of lhs_contracting_dimensions and "
        "rhs_contracting_dimensions must be the same.");
    TF_LITE_ENSURE_MSG(
        context,
        HasUniqueDimensions(params->lhs_batching_dimensions, lhsb_size_,
                            params->lhs_contracting_dimensions, lhsc_size_),
        "'stablehlo.dot_general' lhs_batching_dimensions and "
        "lhs_contracting_dimensions must have unique dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        HasUniqueDimensions(params->rhs_batching_dimensions, rhsb_size_,
                            params->rhs_contracting_dimensions, rhsc_size_),
        "'stablehlo.dot_general' rhs_batching_dimensions and "
        "rhs_contracting_dimensions must have unique dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        !HasInvalidDimension(params->lhs_batching_dimensions, lhsb_size_,
                             lhs_rank_),
        "'stablehlo.dot_general' has invalid lhs_batching_dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        !HasInvalidDimension(params->lhs_contracting_dimensions, lhsc_size_,
                             lhs_rank_),
        "'stablehlo.dot_general' has invalid lhs_contracting_dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        !HasInvalidDimension(params->rhs_batching_dimensions, rhsb_size_,
                             rhs_rank_),
        "'stablehlo.dot_general' has invalid rhs_batching_dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        !HasInvalidDimension(params->rhs_contracting_dimensions, rhsc_size_,
                             rhs_rank_),
        "'stablehlo.dot_general' has invalid rhs_contracting_dimensions.");
    TF_LITE_ENSURE_MSG(
        context,
        HasSameDimensionSize(lhs, rhs, params->lhs_batching_dimensions,
                             params->rhs_batching_dimensions, lhsb_size_),
        "'stablehlo.dot_general' lhs and rhs tensors should have the same "
        "batch dimension size.");
    TF_LITE_ENSURE_MSG(
        context,
        HasSameDimensionSize(lhs, rhs, params->lhs_contracting_dimensions,
                             params->rhs_contracting_dimensions, lhsc_size_),
        "'stablehlo.dot_general' lhs and rhs tensors should have the same "
        "contracting dimension size.");
    if (dequantize::IsQuantizedPerChannel(lhs)) {
      TF_LITE_ENSURE_MSG(context, false,
                         "'stablehlo.dot_general' lhs can't be per-axis "
                         "quantized");
    }
    if (!IsQuantized(lhs) && !IsQuantized(rhs)) {
      TF_LITE_ENSURE_MSG(context, lhs->type == rhs->type,
                         "'stablehlo.dot_general' non-quantized lhs and rhs "
                         "tensors should have the same type");
    }
    if (IsQuantized(lhs) || IsQuantized(rhs) || IsQuantized(output)) {
      TF_LITE_ENSURE_MSG(
          context, IsQuantized(lhs) && IsQuantized(rhs) && IsQuantized(output),
          "'stablehlo.dot_general' if lhs is quantized then rhs and output "
          "tensors must also be quantized");
      if (!dequantize::IsQuantizedPerChannel(rhs)) {
        TF_LITE_ENSURE_MSG(
            context, !dequantize::IsQuantizedPerChannel(output),
            "'stablehlo.dot_general' if lhs and rhs are per tensor quantized "
            "then output tensor must also be per tensor quantized");
        TF_LITE_ENSURE_MSG(
            context, rhs->params.zero_point == 0,
            "'stablehlo.dot_general' rhs per-tensor zero point must be 0.");
      }
      if (dequantize::IsQuantizedPerChannel(rhs)) {
        const auto* affine_quantization =
            reinterpret_cast<TfLiteAffineQuantization*>(
                rhs->quantization.params);
        for (int i = 0; i < affine_quantization->zero_point->size; ++i) {
          TF_LITE_ENSURE_MSG(
              context, affine_quantization->zero_point->data[i] == 0,
              "'stablehlo.dot_general' rhs per-axis zero point must be 0.");
        }
        TF_LITE_ENSURE_MSG(
            context,
            !ArrayContains(params->rhs_contracting_dimensions, rhsc_size_,
                           affine_quantization->quantized_dimension),
            "'stablehlo.dot_general' if rhs is per-axis quantized then "
            "quantization dimension of rhs must not be in "
            "rhs_contracting_dimensions.");
      }
    }
    return TfLiteStatus::kTfLiteOk;
  }

  TfLiteIntArrayUniquePtr GetOutputShape(
      const TfLiteStablehloDotGeneralParams* params, const TfLiteTensor* lhs,
      const TfLiteTensor* rhs) {
    lhs_result_dims_ = CalculateResultDimensions(
        lhs_rank_, params->lhs_batching_dimensions, lhsb_size_,
        params->lhs_contracting_dimensions, lhsc_size_);
    rhs_result_dims_ = CalculateResultDimensions(
        rhs_rank_, params->rhs_batching_dimensions, rhsb_size_,
        params->rhs_contracting_dimensions, rhsc_size_);
    num_lhs_result_dims_ = lhs_result_dims_.size();
    num_rhs_result_dims_ = rhs_result_dims_.size();
    output_rank_ = lhsb_size_ + num_lhs_result_dims_ + num_rhs_result_dims_;

    TfLiteIntArrayUniquePtr output = TfLiteIntArrayUniquePtr(
        TfLiteIntArrayCreate(output_rank_), &TfLiteIntArrayFree);
    // output shape calculation
    for (int i = 0; i < lhsb_size_; ++i) {
      output->data[i] = lhs->dims->data[params->lhs_batching_dimensions[i]];
    }
    for (int i = 0; i < num_lhs_result_dims_; ++i) {
      output->data[i + lhsb_size_] = lhs->dims->data[lhs_result_dims_[i]];
    }
    for (int i = 0; i < num_rhs_result_dims_; ++i) {
      output->data[i + lhsb_size_ + num_lhs_result_dims_] =
          rhs->dims->data[rhs_result_dims_[i]];
    }
    return output;
  }

  void Setup(TfLiteContext* context, TfLiteNode* node,
             const TfLiteStablehloDotGeneralParams* params,
             const TfLiteTensor* lhs, const TfLiteTensor* rhs,
             TfLiteTensor* output) {
    // prepare transpose and reshape tensors
    for (int i = 0; i < lhsb_size_; ++i) {
      newaxes_lhs_.push_back(params->lhs_batching_dimensions[i]);
    }
    for (int i = 0; i < num_lhs_result_dims_; ++i) {
      newaxes_lhs_.push_back(lhs_result_dims_[i]);
    }
    for (int i = 0; i < lhsc_size_; ++i) {
      newaxes_lhs_.push_back(params->lhs_contracting_dimensions[i]);
    }
    for (int i = 0; i < rhsb_size_; ++i) {
      newaxes_rhs_.push_back(params->rhs_batching_dimensions[i]);
    }
    for (int i = 0; i < num_rhs_result_dims_; ++i) {
      newaxes_rhs_.push_back(rhs_result_dims_[i]);
    }
    for (int i = 0; i < rhsc_size_; ++i) {
      newaxes_rhs_.push_back(params->rhs_contracting_dimensions[i]);
    }
    // check if lhs reshape is required
    if (lhs_rank_ == 3 && lhsb_size_ == 1 && lhsc_size_ == 1) {
      do_lhs_reshape_ = false;
    } else {
      int64_t dim = 1;
      if (lhsb_size_ == 0) {
        newshape_lhs_[0] = 1;
      } else {
        for (int i = 0; i < lhsb_size_; ++i) {
          dim *= lhs->dims->data[params->lhs_batching_dimensions[i]];
        }
        newshape_lhs_[0] = dim;
      }
      dim = 1;
      for (int i = 0; i < num_lhs_result_dims_; ++i) {
        dim *= lhs->dims->data[lhs_result_dims_[i]];
      }
      newshape_lhs_[1] = dim;
      dim = 1;
      for (int i = 0; i < lhsc_size_; ++i) {
        dim *= lhs->dims->data[params->lhs_contracting_dimensions[i]];
      }
      newshape_lhs_[2] = dim;
      do_lhs_reshape_ = true;
    }
    // check if rhs reshape is required
    if (rhs_rank_ == 3 && rhsb_size_ == 1 && rhsc_size_ == 1) {
      do_rhs_reshape_ = false;
    } else {
      int64_t dim = 1;
      if (rhsb_size_ == 0) {
        newshape_rhs_[0] = 1;
      } else {
        for (int i = 0; i < rhsb_size_; ++i) {
          dim *= rhs->dims->data[params->rhs_batching_dimensions[i]];
        }
        newshape_rhs_[0] = dim;
      }
      dim = 1;
      for (int i = 0; i < num_rhs_result_dims_; ++i) {
        dim *= rhs->dims->data[rhs_result_dims_[i]];
      }
      newshape_rhs_[1] = dim;
      dim = 1;
      for (int i = 0; i < rhsc_size_; ++i) {
        dim *= rhs->dims->data[params->rhs_contracting_dimensions[i]];
      }
      newshape_rhs_[2] = dim;
      do_rhs_reshape_ = true;
    }
    // lhs_transpose
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(5);
    node->temporaries->data[0] = scratch_tensor_index;
    TfLiteTensor* lhs_transpose;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/0, &lhs_transpose));
    TfLiteIntArray* lhs_transpose_shape = TfLiteIntArrayCreate(lhs_rank_);
    for (int i = 0; i < lhs_rank_; ++i) {
      lhs_transpose_shape->data[i] = lhs->dims->data[newaxes_lhs_[i]];
    }
    lhs_transpose->type = lhs->type;
    lhs_transpose->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, lhs_transpose,
                                                     lhs_transpose_shape));
    // rhs transpose
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* rhs_transpose;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &rhs_transpose));
    TfLiteIntArray* rhs_transpose_shape = TfLiteIntArrayCreate(rhs_rank_);
    for (int i = 0; i < rhs_rank_; ++i) {
      rhs_transpose_shape->data[i] = rhs->dims->data[newaxes_rhs_[i]];
    }
    rhs_transpose->type = rhs->type;
    rhs_transpose->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rhs_transpose,
                                                     rhs_transpose_shape));
    // quantize prepare
    TfLiteIntArray* lhs_dequantize_shape = TfLiteIntArrayCreate(lhs_rank_);
    for (int i = 0; i < lhs_rank_; ++i) {
      lhs_dequantize_shape->data[i] = lhs->dims->data[i];
    }
    node->temporaries->data[2] = scratch_tensor_index + 2;
    TfLiteTensor* lhs_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &lhs_dequantize));
    lhs_dequantize->type = kTfLiteFloat32;
    lhs_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, lhs_dequantize,
                                                     lhs_dequantize_shape));

    TfLiteIntArray* rhs_dequantize_shape = TfLiteIntArrayCreate(rhs_rank_);
    for (int i = 0; i < rhs_rank_; ++i) {
      rhs_dequantize_shape->data[i] = rhs->dims->data[i];
    }
    node->temporaries->data[3] = scratch_tensor_index + 3;
    TfLiteTensor* rhs_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &rhs_dequantize));
    rhs_dequantize->type = kTfLiteFloat32;
    rhs_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rhs_dequantize,
                                                     rhs_dequantize_shape));

    TfLiteIntArray* output_quantize_shape = TfLiteIntArrayCreate(output_rank_);
    for (int i = 0; i < output_rank_; ++i) {
      output_quantize_shape->data[i] = output->dims->data[i];
    }
    node->temporaries->data[4] = scratch_tensor_index + 4;
    TfLiteTensor* output_quantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/4,
                                                &output_quantize));
    output_quantize->type = kTfLiteFloat32;
    output_quantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_quantize,
                                                     output_quantize_shape));
    return kTfLiteOk;
  }

  template <typename DataType>
  TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                        TfLiteTensor* output) {
    // lhs_transpose
    RuntimeShape lhs_transposed_shape(GetTensorShape(lhs));
    RuntimeShape lhs_shape(GetTensorShape(lhs));
    for (int i = 0; i < lhs_shape.DimensionsCount(); ++i) {
      lhs_transposed_shape.SetDim(i, lhs_shape.Dims(newaxes_lhs_[i]));
    }
    TransposeParams lhs_params;
    lhs_params.perm_count = lhs_rank_;
    for (int i = 0; i < NumDimensions(lhs); ++i) {
      lhs_params.perm[i] = newaxes_lhs_[i];
    }
    TfLiteTensor* lhs_transpose = GetTemporary(context, node, 0);
    optimized_ops::Transpose(lhs_params, lhs_shape,
                             GetTensorData<DataType>(lhs), lhs_transposed_shape,
                             GetTensorData<DataType>(lhs_transpose));
    // rhs_transpose
    RuntimeShape rhs_transposed_shape(GetTensorShape(rhs));
    RuntimeShape rhs_shape(GetTensorShape(rhs));
    for (int i = 0; i < rhs_shape.DimensionsCount(); ++i) {
      rhs_transposed_shape.SetDim(i, rhs_shape.Dims(newaxes_rhs_[i]));
    }
    TransposeParams rhs_params;
    rhs_params.perm_count = rhs_rank_;
    for (int i = 0; i < NumDimensions(rhs); ++i) {
      rhs_params.perm[i] = newaxes_rhs_[i];
    }
    TfLiteTensor* rhs_transpose = GetTemporary(context, node, 1);
    optimized_ops::Transpose(rhs_params, rhs_shape,
                             GetTensorData<DataType>(rhs), rhs_transposed_shape,
                             GetTensorData<DataType>(rhs_transpose));
    // lhs reshape
    if (do_lhs_reshape_) {
      const int lhs_reshape_size = 3;
      TfLiteIntArray* lhs_reshape = TfLiteIntArrayCreate(lhs_reshape_size);
      for (int i = 0; i < lhs_reshape_size; ++i) {
        lhs_reshape->data[i] = newshape_lhs_[i];
      }
      lhs_transpose->dims = lhs_reshape;
    }
    // rhs reshape
    if (do_rhs_reshape_) {
      const int rhs_reshape_size = 3;
      TfLiteIntArray* rhs_reshape = TfLiteIntArrayCreate(rhs_reshape_size);
      for (int i = 0; i < rhs_reshape_size; ++i) {
        rhs_reshape->data[i] = newshape_rhs_[i];
      }
      rhs_transpose->dims = rhs_reshape;
    }
    // matrix multiplication using eigen
    DataType* lhs_data = GetTensorData<DataType>(lhs_transpose);
    DataType* rhs_data = GetTensorData<DataType>(rhs_transpose);
    DataType* output_data = GetTensorData<DataType>(output);

    const int batch_size = lhs_transpose->dims->data[0];
    const int n = lhs_transpose->dims->data[1];
    const int m = rhs_transpose->dims->data[1];
    const int p = lhs_transpose->dims->data[2];
    const int output_batch_size = n * m;
    const int lhs_batch_size = n * p;
    const int rhs_batch_size = m * p;

    using EigenMatrixMapRowMajorConst =
        Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>;
    using EigenMatrixMapColMajorConst =
        Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>>;
    using EigenMatrixMapRowMajorMutable =
        Eigen::Map<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>;

    for (int batch = 0; batch < batch_size; ++batch) {
      EigenMatrixMapRowMajorConst eigen_lhs(lhs_data + batch * lhs_batch_size,
                                            n, p);
      EigenMatrixMapColMajorConst eigen_rhs(rhs_data + batch * rhs_batch_size,
                                            p, m);
      EigenMatrixMapRowMajorMutable eigen_dst(
          output_data + batch * output_batch_size, n, m);
      if (m == 1) {
        eigen_dst.col(0).noalias() = eigen_lhs * eigen_rhs.col(0);
      } else if (n == 1) {
        eigen_dst.row(0).noalias() = eigen_lhs.row(0) * eigen_rhs;
      } else {
        eigen_dst.noalias() = eigen_lhs * eigen_rhs;
      }
    }
    return TfLiteStatus::kTfLiteOk;
  }

  template <typename DataType>
  TfLiteStatus EvalQuantize(TfLiteContext* context, TfLiteNode* node,
                            const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                            TfLiteTensor* output) {
    TfLiteTensor* lhs_dequantize = GetTemporary(context, node, 2);
    TfLiteTensor* rhs_dequantize = GetTemporary(context, node, 3);
    TfLiteTensor* output_dequantize = GetTemporary(context, node, 4);

    dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
        context, node, lhs, lhs_dequantize);
    dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
        context, node, rhs, rhs_dequantize);
    EvalImpl<float>(context, node, lhs_dequantize, rhs_dequantize,
                    output_dequantize);

    RuntimeShape output_shape(GetTensorShape(output));
    RuntimeShape output_dequantize_shape(GetTensorShape(output_dequantize));
    if (dequantize::IsQuantizedPerChannel(output)) {
      const auto* quantization_params =
          reinterpret_cast<const TfLiteAffineQuantization*>(
              output->quantization.params);
      PerChannelQuantizationParams per_channel_op_params;
      per_channel_op_params.quantized_dimension =
          quantization_params->quantized_dimension;
      per_channel_op_params.scale = quantization_params->scale->data;
      per_channel_op_params.zero_point = quantization_params->zero_point->data;
      reference_ops::PerChannelQuantize(
          per_channel_op_params, output_dequantize_shape,
          GetTensorData<float>(output_dequantize), output_shape,
          GetTensorData<DataType>(output));
    } else {
      tflite::QuantizationParams op_params;
      op_params.zero_point = output->params.zero_point;
      op_params.scale = output->params.scale;
      optimized_ops::AffineQuantize<DataType>(
          op_params, output_dequantize_shape,
          GetTensorData<float>(output_dequantize), output_shape,
          GetTensorData<DataType>(output));
    }
    return kTfLiteOk;
  }

 private:
  int lhs_rank_;
  int rhs_rank_;
  int lhsb_size_;
  int rhsb_size_;
  int lhsc_size_;
  int rhsc_size_;
  int output_rank_;
  int num_lhs_result_dims_;
  int num_rhs_result_dims_;
  std::vector<int64_t> lhs_result_dims_;
  std::vector<int64_t> rhs_result_dims_;
  std::vector<int64_t> newaxes_lhs_;
  std::vector<int64_t> newaxes_rhs_;
  int64_t newshape_lhs_[3];
  int64_t newshape_rhs_[3];
  bool do_lhs_reshape_;
  bool do_rhs_reshape_;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  DotGeneralData* dot_general_data = new DotGeneralData();
  context->AddTensors(context, 5, &dot_general_data->scratch_tensor_index);
  return dot_general_data;
}

void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<DotGeneralData*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* lhs;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, DotGeneralData::kLhsTensor, &lhs));
  const TfLiteTensor* rhs;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, DotGeneralData::kRhsTensor, &rhs));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, DotGeneralData::kOutputTensor, &output));

  DotGeneralData& dot_general_data =
      *reinterpret_cast<DotGeneralData*>(node->user_data);
  const TfLiteStablehloDotGeneralParams* params =
      reinterpret_cast<TfLiteStablehloDotGeneralParams*>(node->builtin_data);
  // Constraint checks Stablehlo specs
  TF_LITE_ENSURE_OK(context, dot_general_data.CheckParameters(
                                 context, params, lhs, rhs, output));
  TfLiteIntArrayUniquePtr result_shape =
      dot_general_data.GetOutputShape(params, lhs, rhs);
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, output, result_shape.release()));
  // prepare lhs and rhs transpose tensor
  dot_general_data.Setup(context, node, params, lhs, rhs, output);
  return TfLiteStatus::kTfLiteOk;
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, DotGeneralData::kLhsTensor, &lhs));
  const TfLiteTensor* rhs;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, DotGeneralData::kRhsTensor, &rhs));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, DotGeneralData::kOutputTensor, &output));

  TfLiteType data_type = lhs->type;
  DotGeneralData& dot_general_data =
      *reinterpret_cast<DotGeneralData*>(node->user_data);
  if (data_type == kTfLiteFloat32) {
    return dot_general_data.EvalImpl<float>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteBFloat16) {
    return dot_general_data.EvalImpl<Eigen::bfloat16>(context, node, lhs, rhs,
                                                      output);
  } else if (data_type == kTfLiteFloat16) {
    return dot_general_data.EvalImpl<Eigen::half>(context, node, lhs, rhs,
                                                  output);
  } else if (data_type == kTfLiteInt8) {
    return dot_general_data.EvalQuantize<int8_t>(context, node, lhs, rhs,
                                                 output);
  } else if (data_type == kTfLiteInt16) {
    return dot_general_data.EvalQuantize<int16_t>(context, node, lhs, rhs,
                                                  output);
  } else if (data_type == kTfLiteInt32) {
    return dot_general_data.EvalImpl<int32_t>(context, node, lhs, rhs, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(DataType: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace stablehlo_dot_general

TfLiteRegistration* Register_STABLEHLO_DOT_GENERAL() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_dot_general::Init,
                                 /*.free=*/stablehlo_dot_general::Free,
                                 /*.prepare=*/stablehlo_dot_general::Prepare,
                                 /*.invoke=*/stablehlo_dot_general::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
