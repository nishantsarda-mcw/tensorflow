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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::Eq;
using testing::FloatEq;
using testing::FloatNear;
using testing::Pointwise;
namespace shlo_ref {

namespace {

using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
template <class T>
struct NonQuantizedkF16DotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF16DotGeneralTest, kF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF16DotGeneralTest, kF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2});
  const Shape shape_rhs({2, 2, 2});
  const Shape shape_r({2, 2, 2});

  Vector<float> lhs_data_float{1.1, 2.2, 3.3, 4.3, 5.5, 6, 7, 8};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {1.31934, 2.63867, 3.96094, 5.16016,
                                       6.60156, 7.20312, 8.39844, 9.60156};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
