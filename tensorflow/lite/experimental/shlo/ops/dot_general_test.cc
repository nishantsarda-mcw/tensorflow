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
struct ConstraintTest : ::testing::Test {};
TYPED_TEST_SUITE(ConstraintTest, IntTestTypes, TestParamNames);

TYPED_TEST(ConstraintTest, InvalidOutputShapeRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2});
  const Shape shape_rhs({2, 2});
  const Shape shape_r({2, 2, 2});
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

  Vector<int64_t> expected_data_int{1, 0, 0, 1, 2, 0, 0, 2};

  const absl::Status status = Prepare(op, lhs, rhs, output_tensor);
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(), "stablehlo.dot_general: Invalid output shape.");
}

template <class T>
struct NonQuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntDotGeneralTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shape_lhs({7, 3, 4});
  const Shape shape_rhs({7, 4});
  const Shape shape_r({7, 3});

  Vector<int64_t> lhs_data_int{
      0,  1,  4,  1,  -2, -3, 0, 0, 6,  -1, 0,  0,  1,  0,  -2, 0,  1,
      3,  4,  -6, 2,  4,  4,  0, 0, -2, -1, 1,  -2, -3, 0,  2,  -3, 0,
      0,  -2, 4,  -7, 2,  2,  0, 4, 2,  0,  -6, 1,  1,  2,  -2, -2, 0,
      -1, -4, -1, 0,  -1, 1,  3, 1, 1,  -4, 0,  0,  1,  -1, 0,  4,  -2,
      0,  5,  0,  -1, 0,  2,  1, 2, -1, 1,  -3, -2, -6, -3, -1, -3};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{2,  0, -1, 4,  -4, 0,  2,  -1, 0, 6,
                               8,  0, -1, -3, -1, -1, -3, 0,  5, 0,
                               -3, 0, 3,  -1, 2,  1,  -2, -3};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
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

  Vector<int64_t> expected_data_int{0,   -4, 12, -8,  10, 0,  -20,
                                    -18, 0,  13, -14, 0,  6,  12,
                                    2,   11, 17, 1,   -6, 11, -4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;
template <class T>
struct NonQuantizedkF32DotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedkF32DotGeneralTest, kF32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4, 3});
  const Shape shape_rhs({3, 6});
  const Shape shape_r({4, 6});

  Vector<StorageT> lhs_data{5.81311798,  2.08485532,  0.151162371, -1.21007407,
                            -1.59476554, 0.846119463, -0.83784312, -0.416278511,
                            1.24929118,  3.46354723,  2.21915126,  3.81866336};
  Vector<StorageT> rhs_data{
      -2.10215521, -1.803730e+00, -7.83739519, 4.36787844,   1.4788357,
      3.10357666,  -4.46420813,   0.879630148, -2.18081808,  -1.95115197,
      -3.56435633, -0.671983778,  -2.76886797, -0.212248296, 2.77085519,
      -1.21441388, -3.28464937,   -4.60568237};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{0};
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

  Vector<StorageT> expected_data = {
      -21.9458523, -8.6834774,  -49.6875458, 21.1395512,   0.668963671,
      15.9442625,  7.32033587,  0.600255728, 15.3061972,   -3.20136595,
      1.11560607,  -6.58085871, 0.160507768, 0.87991172,   10.9359407,
      -4.36453056, -3.85875082, -8.07441616, -27.7610416,  -5.10577631,
      -21.4037914, 6.1610136,   -15.3307991, -8.329400e+00};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatNear(1e-5), expected_data));
}

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

template <class T>
struct QuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(QuantizedIntDotGeneralTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({4, 3});
  const Shape shape_rhs({3});
  const Shape shape_r({4});

  Vector<StorageT> lhs_data =
      Vector<StorageT>{0, 0, 2, 0, 1, 2, 4, 2, 0, 1, 2, 6};
  Vector<StorageT> rhs_data = Vector<StorageT>{1, 1, 0};
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  Vector<StorageT> output_data(shape_r.NumElements());

  const ExpressedT scale = static_cast<ExpressedT>(1.2);
  const StorageT zero_point = static_cast<StorageT>(-1);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  const QuantizedElementTypePerTensor tensor_type_rhs =
      QuantizedElementTypePerTensor(TypeParam::kStorage, 0,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerTensorTensorType{.shape = shape_rhs,
                                           .element_type = tensor_type_rhs},
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

  Vector<float> expected_data = Vector<float>{2.88, 4.32, 11.531, 7.2};
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
