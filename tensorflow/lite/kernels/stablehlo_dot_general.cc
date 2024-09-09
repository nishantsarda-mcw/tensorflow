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
static constexpr int kMaxTemporaryTensors = 2;

struct DotGeneralData {
 public:
  enum { kLhsTensor, kRhsTensor };
  enum { kOutputTensor };
  // The index of the temporary tensors
  int scratch_tensor_index;
  std::vector<int64_t> lhs_result_dims;
  std::vector<int64_t> rhs_result_dims;
  std::vector<int64_t> newaxes_lhs;
  std::vector<int64_t> newaxes_rhs;
  int64_t newshape_lhs[3];
  int64_t newshape_rhs[3];
  bool do_lhs_reshape;
  bool do_rhs_reshape;
};

using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;

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

TfLiteStatus CheckParameters(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteStablehloDotGeneralParams* params,
                             const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                             const TfLiteTensor* output) {
  TF_LITE_ENSURE_MSG(context, params->num_precision_configs == 2,
                     "'stablehlo.dot_general' size of precision_config "
                     "parameter must be two.");
  TF_LITE_ENSURE_MSG(
      context,
      params->num_lhs_batching_dimensions ==
          params->num_rhs_batching_dimensions,
      "'stablehlo.dot_general' size of lhs_batching_dimensions and "
      "rhs_batching_dimensions must be the same.");
  TF_LITE_ENSURE_MSG(
      context,
      params->num_lhs_contracting_dimensions ==
          params->num_rhs_contracting_dimensions,
      "'stablehlo.dot_general' size of lhs_contracting_dimensions and "
      "rhs_contracting_dimensions must be the same.");
  TF_LITE_ENSURE_MSG(
      context,
      HasUniqueDimensions(params->lhs_batching_dimensions,
                          params->num_lhs_batching_dimensions,
                          params->lhs_contracting_dimensions,
                          params->num_lhs_contracting_dimensions),
      "'stablehlo.dot_general' lhs_batching_dimensions and "
      "lhs_contracting_dimensions must have unique dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      HasUniqueDimensions(params->rhs_batching_dimensions,
                          params->num_rhs_batching_dimensions,
                          params->rhs_contracting_dimensions,
                          params->num_rhs_contracting_dimensions),
      "'stablehlo.dot_general' rhs_batching_dimensions and "
      "rhs_contracting_dimensions must have unique dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      !HasInvalidDimension(params->lhs_batching_dimensions,
                           params->num_lhs_batching_dimensions,
                           NumDimensions(lhs)),
      "'stablehlo.dot_general' has invalid lhs_batching_dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      !HasInvalidDimension(params->lhs_contracting_dimensions,
                           params->num_lhs_contracting_dimensions,
                           NumDimensions(lhs)),
      "'stablehlo.dot_general' has invalid lhs_contracting_dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      !HasInvalidDimension(params->rhs_batching_dimensions,
                           params->num_rhs_batching_dimensions,
                           NumDimensions(rhs)),
      "'stablehlo.dot_general' has invalid rhs_batching_dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      !HasInvalidDimension(params->rhs_contracting_dimensions,
                           params->num_rhs_contracting_dimensions,
                           NumDimensions(rhs)),
      "'stablehlo.dot_general' has invalid rhs_contracting_dimensions.");
  TF_LITE_ENSURE_MSG(
      context,
      HasSameDimensionSize(lhs, rhs, params->lhs_batching_dimensions,
                           params->rhs_batching_dimensions,
                           params->num_lhs_batching_dimensions),
      "'stablehlo.dot_general' lhs and rhs tensors should have the same "
      "batch dimension size.");
  TF_LITE_ENSURE_MSG(
      context,
      HasSameDimensionSize(lhs, rhs, params->lhs_contracting_dimensions,
                           params->rhs_contracting_dimensions,
                           params->num_lhs_contracting_dimensions),
      "'stablehlo.dot_general' lhs and rhs tensors should have the same "
      "contracting dimension size.");
  return TfLiteStatus::kTfLiteOk;
}

static TfLiteIntArrayUniquePtr GetResultShape(
    TfLiteNode* node, const TfLiteStablehloDotGeneralParams* params,
    const TfLiteTensor* lhs, const TfLiteTensor* rhs) {
  DotGeneralData& opdata = *reinterpret_cast<DotGeneralData*>(node->user_data);

  const int lhs_rank = lhs->dims->size;
  const int rhs_rank = rhs->dims->size;
  const int lhsb_size = params->num_lhs_batching_dimensions;
  const int rhsb_size = params->num_rhs_batching_dimensions;
  const int lhsc_size = params->num_lhs_contracting_dimensions;
  const int rhsc_size = params->num_rhs_contracting_dimensions;

  opdata.lhs_result_dims = CalculateResultDimensions(
      lhs_rank, params->lhs_batching_dimensions, lhsb_size,
      params->lhs_contracting_dimensions, lhsc_size);
  opdata.rhs_result_dims = CalculateResultDimensions(
      rhs_rank, params->rhs_batching_dimensions, rhsb_size,
      params->rhs_contracting_dimensions, rhsc_size);
  int result_rank =
      lhsb_size + opdata.lhs_result_dims.size() + opdata.rhs_result_dims.size();
  TfLiteIntArrayUniquePtr result = TfLiteIntArrayUniquePtr(
      TfLiteIntArrayCreate(result_rank), &TfLiteIntArrayFree);
  // output shape calculation
  for (int i = 0; i < lhsb_size; ++i) {
    result->data[i] = lhs->dims->data[params->lhs_batching_dimensions[i]];
  }
  for (int i = 0; i < opdata.lhs_result_dims.size(); ++i) {
    result->data[i + lhsb_size] = lhs->dims->data[opdata.lhs_result_dims[i]];
  }
  for (int i = 0; i < opdata.rhs_result_dims.size(); ++i) {
    result->data[i + lhsb_size + opdata.lhs_result_dims.size()] =
        rhs->dims->data[opdata.rhs_result_dims[i]];
  }
  return result;
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteStablehloDotGeneralParams* params,
                                const TfLiteTensor* lhs,
                                const TfLiteTensor* rhs, TfLiteTensor* output) {
  DotGeneralData& opdata = *reinterpret_cast<DotGeneralData*>(node->user_data);

  context->AddTensors(context, kMaxTemporaryTensors,
                      &opdata.scratch_tensor_index);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kMaxTemporaryTensors);

  const int lhs_rank = NumDimensions(lhs);
  const int rhs_rank = NumDimensions(rhs);
  const int output_rank = NumDimensions(output);
  const int lhsb_size = params->num_lhs_batching_dimensions;
  const int rhsb_size = params->num_rhs_batching_dimensions;
  const int lhsc_size = params->num_lhs_contracting_dimensions;
  const int rhsc_size = params->num_rhs_contracting_dimensions;
  const int num_lhs_result_dims = opdata.lhs_result_dims.size();
  const int num_rhs_result_dims = opdata.rhs_result_dims.size();
  // prepare transpose tensors - putting batching dims at the start and
  // contracting dims at the end
  for (int i = 0; i < lhsb_size; ++i) {
    opdata.newaxes_lhs.push_back(params->lhs_batching_dimensions[i]);
  }
  for (int i = 0; i < num_lhs_result_dims; ++i) {
    opdata.newaxes_lhs.push_back(opdata.lhs_result_dims[i]);
  }
  for (int i = 0; i < lhsc_size; ++i) {
    opdata.newaxes_lhs.push_back(params->lhs_contracting_dimensions[i]);
  }
  for (int i = 0; i < rhsb_size; ++i) {
    opdata.newaxes_rhs.push_back(params->rhs_batching_dimensions[i]);
  }
  for (int i = 0; i < num_rhs_result_dims; ++i) {
    opdata.newaxes_rhs.push_back(opdata.rhs_result_dims[i]);
  }
  for (int i = 0; i < rhsc_size; ++i) {
    opdata.newaxes_rhs.push_back(params->rhs_contracting_dimensions[i]);
  }
  // check if lhs reshape is required. If lhs rank is 3 and lhsb_size == 1 and
  // lhsc_size == 1 then no reshape is required
  if (lhs_rank == 3 && lhsb_size == 1 && lhsc_size == 1) {
    opdata.do_lhs_reshape = false;
  } else {
    int64_t dim = 1;
    if (lhsb_size == 0) {
      opdata.newshape_lhs[0] = 1;
    } else {
      for (int i = 0; i < lhsb_size; ++i) {
        dim *= lhs->dims->data[params->lhs_batching_dimensions[i]];
      }
      opdata.newshape_lhs[0] = dim;
    }
    dim = 1;
    for (int i = 0; i < num_lhs_result_dims; ++i) {
      dim *= lhs->dims->data[opdata.lhs_result_dims[i]];
    }
    opdata.newshape_lhs[1] = dim;
    dim = 1;
    for (int i = 0; i < lhsc_size; ++i) {
      dim *= lhs->dims->data[params->lhs_contracting_dimensions[i]];
    }
    opdata.newshape_lhs[2] = dim;
    opdata.do_lhs_reshape = true;
  }
  // check if rhs reshape is required
  if (rhs_rank == 3 && rhsb_size == 1 && rhsc_size == 1) {
    opdata.do_rhs_reshape = false;
  } else {
    int64_t dim = 1;
    if (rhsb_size == 0) {
      opdata.newshape_rhs[0] = 1;
    } else {
      for (int i = 0; i < rhsb_size; ++i) {
        dim *= rhs->dims->data[params->rhs_batching_dimensions[i]];
      }
      opdata.newshape_rhs[0] = dim;
    }
    dim = 1;
    for (int i = 0; i < num_rhs_result_dims; ++i) {
      dim *= rhs->dims->data[opdata.rhs_result_dims[i]];
    }
    opdata.newshape_rhs[1] = dim;
    dim = 1;
    for (int i = 0; i < rhsc_size; ++i) {
      dim *= rhs->dims->data[params->rhs_contracting_dimensions[i]];
    }
    opdata.newshape_rhs[2] = dim;
    opdata.do_rhs_reshape = true;
  }
  // lhs_transpose prepare
  node->temporaries->data[0] = opdata.scratch_tensor_index;
  TfLiteTensor* lhs_transpose;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &lhs_transpose));
  TfLiteIntArray* lhs_transpose_shape = TfLiteIntArrayCreate(lhs_rank);
  for (int i = 0; i < lhs_rank; ++i) {
    lhs_transpose_shape->data[i] = lhs->dims->data[opdata.newaxes_lhs[i]];
  }
  lhs_transpose->type = lhs->type;
  lhs_transpose->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, lhs_transpose,
                                                   lhs_transpose_shape));
  // lhs reshape - to 3D tensor
  if (opdata.do_lhs_reshape) {
    const int lhs_reshape_size = 3;
    TfLiteIntArray* lhs_reshape = TfLiteIntArrayCreate(lhs_reshape_size);
    for (int i = 0; i < lhs_reshape_size; ++i) {
      lhs_reshape->data[i] = opdata.newshape_lhs[i];
    }
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, lhs_transpose, lhs_reshape));
  }
  // rhs_transpose prepare
  node->temporaries->data[1] = opdata.scratch_tensor_index + 1;
  TfLiteTensor* rhs_transpose;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &rhs_transpose));
  TfLiteIntArray* rhs_transpose_shape = TfLiteIntArrayCreate(rhs_rank);
  for (int i = 0; i < rhs_rank; ++i) {
    rhs_transpose_shape->data[i] = rhs->dims->data[opdata.newaxes_rhs[i]];
  }
  rhs_transpose->type = rhs->type;
  rhs_transpose->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rhs_transpose,
                                                   rhs_transpose_shape));
  // rhs reshape - to 3D tensor
  if (opdata.do_rhs_reshape) {
    const int rhs_reshape_size = 3;
    TfLiteIntArray* rhs_reshape = TfLiteIntArrayCreate(rhs_reshape_size);
    for (int i = 0; i < rhs_reshape_size; ++i) {
      rhs_reshape->data[i] = opdata.newshape_rhs[i];
    }
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, rhs_transpose, rhs_reshape));
  }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                      TfLiteTensor* output) {
  DotGeneralData& opdata = *reinterpret_cast<DotGeneralData*>(node->user_data);

  const int lhs_rank = NumDimensions(lhs);
  const int rhs_rank = NumDimensions(rhs);
  // lhs transpose
  RuntimeShape lhs_transposed_shape(GetTensorShape(lhs));
  RuntimeShape lhs_shape(GetTensorShape(lhs));
  for (int i = 0; i < lhs_shape.DimensionsCount(); ++i) {
    lhs_transposed_shape.SetDim(i, lhs_shape.Dims(opdata.newaxes_lhs[i]));
  }
  TransposeParams lhs_params;
  lhs_params.perm_count = lhs_rank;
  for (int i = 0; i < lhs_rank; ++i) {
    lhs_params.perm[i] = opdata.newaxes_lhs[i];
  }
  TfLiteTensor* lhs_transpose = GetTemporary(context, node, 0);
  optimized_ops::Transpose(lhs_params, lhs_shape, GetTensorData<DataType>(lhs),
                           lhs_transposed_shape,
                           GetTensorData<DataType>(lhs_transpose));
  // rhs transpose
  RuntimeShape rhs_transposed_shape(GetTensorShape(rhs));
  RuntimeShape rhs_shape(GetTensorShape(rhs));
  for (int i = 0; i < rhs_shape.DimensionsCount(); ++i) {
    rhs_transposed_shape.SetDim(i, rhs_shape.Dims(opdata.newaxes_rhs[i]));
  }
  TransposeParams rhs_params;
  rhs_params.perm_count = rhs_rank;
  for (int i = 0; i < rhs_rank; ++i) {
    rhs_params.perm[i] = opdata.newaxes_rhs[i];
  }
  TfLiteTensor* rhs_transpose = GetTemporary(context, node, 1);
  optimized_ops::Transpose(rhs_params, rhs_shape, GetTensorData<DataType>(rhs),
                           rhs_transposed_shape,
                           GetTensorData<DataType>(rhs_transpose));

  // matrix multiplication using eigen
  const DataType* lhs_data = GetTensorData<DataType>(lhs_transpose);
  const DataType* rhs_data = GetTensorData<DataType>(rhs_transpose);
  DataType* output_data = GetTensorData<DataType>(output);

  const int batch_size = lhs_transpose->dims->data[0];
  const int n = lhs_transpose->dims->data[1];
  const int p = lhs_transpose->dims->data[2];
  const int m = rhs_transpose->dims->data[1];
  const int output_batch_size = n * m;
  const int lhs_batch_size = n * p;
  const int rhs_batch_size = m * p;

  using EigenMatrixMapRowMajorConst =
      Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>;
  using EigenMatrixMapColMajorConst =
      Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>;
  using EigenMatrixMapRowMajorMutable = Eigen::Map<
      Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  for (int batch = 0; batch < batch_size; ++batch) {
    EigenMatrixMapRowMajorConst eigen_lhs(lhs_data + batch * lhs_batch_size, n,
                                          p);
    EigenMatrixMapColMajorConst eigen_rhs(rhs_data + batch * rhs_batch_size, p,
                                          m);
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

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  DotGeneralData* dot_general_data = new DotGeneralData();
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
  TF_LITE_ENSURE_EQ(context, lhs->type, rhs->type);

  const TfLiteStablehloDotGeneralParams* params =
      reinterpret_cast<TfLiteStablehloDotGeneralParams*>(node->builtin_data);
  // Constraint checks as per StableHLO specs
  TF_LITE_ENSURE_OK(context,
                    CheckParameters(context, node, params, lhs, rhs, output));
  // Output tensor setup
  TfLiteIntArrayUniquePtr result_shape = GetResultShape(node, params, lhs, rhs);
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, output, result_shape.release()));
  // Prepare temporary tensors for lhs and rhs transpose
  TF_LITE_ENSURE_OK(
      context, PrepareTemporaries(context, node, params, lhs, rhs, output));

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
  if (data_type == kTfLiteInt8) {
    return EvalImpl<int8_t>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteBFloat16) {
    return EvalImpl<Eigen::bfloat16>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteFloat16) {
    return EvalImpl<Eigen::half>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteFloat32) {
    return EvalImpl<float>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteInt16) {
    return EvalImpl<int16_t>(context, node, lhs, rhs, output);
  } else if (data_type == kTfLiteInt32) {
    return EvalImpl<int32_t>(context, node, lhs, rhs, output);
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
