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

template <class T>
struct NonQuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntDotGeneralTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2});
  const Shape shape_rhs({2, 2});
  const Shape shape_r({2, 2, 2, 2});
  Vector<int64_t> lhs_data_int{1, 2, 3, 4};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{1, 0, 0, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{};
  Vector<Axis> rhsc_dim{};
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

  Vector<int64_t> expected_data_int{1, 0, 0, 1, 2, 0, 0, 2,
                                    3, 0, 0, 3, 4, 0, 0, 4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
template <class T>
struct NonQuantizedkF16DotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedkF16DotGeneralTest, kF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF16DotGeneralTest, kF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2, 2});
  const Shape shape_rhs({2, 2, 2, 2});
  const Shape shape_r({2, 2, 2, 2});

  Vector<float> lhs_data_float{1.1,  2.2,   3.3,   4.3,   5.5,   6,   7,   8,
                               11.1, 12.22, 33.33, 44.32, 15.15, 6.6, 7.3, 8.1};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2,
                               1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0, 3};
  Vector<Axis> rhsb_dim{0, 3};
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
  Vector<float> expected_data_float = {
      1.319, 1.319, 6.6,   6.6,   5.16,   5.16,   9.6,    9.6,
      13.32, 13.32, 18.18, 18.18, 53.184, 53.184, 9.7265, 9.7265};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(QuantizedIntDotGeneralTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({1, 2, 2});
  const Shape shape_rhs({1, 2, 2});
  const Shape shape_r({1, 2, 2});

  Vector<StorageT> lhs_data = Vector<StorageT>{10, 8, 1, 2};
  Vector<StorageT> rhs_data = Vector<StorageT>{1, 0, 1, 1};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  const ExpressedT scale = static_cast<ExpressedT>(2);
  const StorageT zero_point = static_cast<StorageT>(0);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{.type = QuantizedPerTensorTensorType{.shape = shape_rhs,
                                                  .element_type = tensor_type},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data = Vector<float>{44, 4, 40, 8};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

}  // namespace
}  // namespace shlo_ref
