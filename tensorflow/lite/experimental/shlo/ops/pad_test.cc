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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
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

  const Shape shape_operand({5, 4});
  const Shape shape_padding_value{};
  const Shape shape_result({7, 4});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 1, 2, 0, 0, 3,
                                4, 0, 0, 5, 6, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{-1, -1, -1, -1, -1, -1, 0, -1, 0, -1,
                                 -1, 1,  -1, 2,  -1, -1, 3, -1, 4, -1,
                                 -1, 5,  -1, 6,  -1, -1, 0, -1};
  Vector<DimensionSize> pad_value{-1};
  Vector<DimensionSize> edge_padding_low{1, -1};
  Vector<DimensionSize> edge_padding_high{1, -1};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  const absl::Status status =
      Prepare(op, operand, padding_value, output_tensor);

  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(), "stablehlo.pad: Invalid output shape.");
}

template <class T>
struct NonQuantizedIntPadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntPadTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({5, 4});
  const Shape shape_padding_value{};
  const Shape shape_result({7, 5});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 1, 2, 0, 0, 3,
                                4, 0, 0, 5, 6, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{-1, -1, -1, -1, -1, -1, 0,  -1, 0,  -1, -1, 1,
                                 -1, 2,  -1, -1, 3,  -1, 4,  -1, -1, 5,  -1, 6,
                                 -1, -1, 0,  -1, 0,  -1, -1, -1, -1, -1, -1};
  Vector<DimensionSize> pad_value{-1};
  Vector<DimensionSize> edge_padding_low{1, -1};
  Vector<DimensionSize> edge_padding_high{1, -1};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({5, 9});

  Vector<StorageT> operand_data{1, 2, 3, 4, 5, 6};
  Vector<StorageT> expected_data{0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Vector<DimensionSize> pad_value{0};
  Vector<DimensionSize> edge_padding_low{0, 1};
  Vector<DimensionSize> edge_padding_high{2, 1};
  Vector<DimensionSize> interior_padding{1, 2};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 1});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{0, 0};
  Vector<DimensionSize> pad_value{0};
  Vector<DimensionSize> edge_padding_low{0, -1};
  Vector<DimensionSize> edge_padding_high{0, -1};
  Vector<DimensionSize> interior_padding{0, 0};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork4) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({4, 7});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Vector<DimensionSize> pad_value{0};
  Vector<DimensionSize> edge_padding_low{1, 2};
  Vector<DimensionSize> edge_padding_high{1, 2};
  Vector<DimensionSize> interior_padding{0, 0};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};
  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

template <class T>
struct NonQuantizedBoolPadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedBoolPadTest, BoolTestType, TestParamNames);

TYPED_TEST(NonQuantizedBoolPadTest, BoolTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({6, 4});

  Vector<StorageT> operand_data{true, true, true, true, true, true};
  Vector<StorageT> expected_data{false, false, false, false, true,  true,
                                 true,  false, false, false, false, false,
                                 true,  true,  true,  false, false, false,
                                 false, false, false, false, false, false};
  Vector<DimensionSize> pad_value{false};
  Vector<DimensionSize> edge_padding_low{1, 0};
  Vector<DimensionSize> edge_padding_high{2, 1};
  Vector<DimensionSize> interior_padding{1, 0};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedBoolPadTest, BoolTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 0});

  Vector<StorageT> operand_data{true, true, true, true, true, true};
  Vector<StorageT> expected_data{};
  Vector<DimensionSize> pad_value{false};
  Vector<DimensionSize> edge_padding_low{0, -2};
  Vector<DimensionSize> edge_padding_high{0, -3};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
template <class T>
struct NonQuantizedkBF16PadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedkBF16PadTest, kBF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkBF16PadTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 7});

  Vector<float> operand_data_int{-1.430510e-04, 5.836480e-04,  9.040830e-04,
                                 -8.163450e-04, -1.152040e-03, 4.291530e-04};
  Vector<StorageT> operand_data(operand_data_int.begin(),
                                operand_data_int.end());
  Vector<float> expected_data_int{
      0.000000e+00,  0.000000e+00, 0.000000e+00, 5.836480e-04, 0.000000e+00,
      0.000000e+00,  0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
      -1.152040e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());
  Vector<StorageT> pad_value{StorageT(0.000000e+00)};
  Vector<DimensionSize> edge_padding_low{0, -2};
  Vector<DimensionSize> edge_padding_high{0, -2};
  Vector<DimensionSize> interior_padding{0, 4};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
template <class T>
struct NonQuantizedkF16PadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedkF16PadTest, kF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF16PadTest, kF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 7});

  Vector<float> operand_data_int{-1.026150e-03, 1.046660e-04,  1.714230e-04,
                                 -1.002310e-03, -5.316730e-04, 4.243850e-04};
  Vector<StorageT> operand_data(operand_data_int.begin(),
                                operand_data_int.end());
  Vector<float> expected_data_int{
      0.000000e+00,  0.000000e+00, 0.000000e+00, 1.046660e-04, 0.000000e+00,
      0.000000e+00,  0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
      -5.316730e-04, 0.000000e+00, 0.000000e+00, 0.000000e+00};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());
  Vector<StorageT> pad_value{StorageT(0.000000e+00)};
  Vector<DimensionSize> edge_padding_low{0, -2};
  Vector<DimensionSize> edge_padding_high{0, -2};
  Vector<DimensionSize> interior_padding{0, 4};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;
template <class T>
struct NonQuantizedkF32PadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedkF32PadTest, kF32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF32PadTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 1});

  Vector<StorageT> operand_data{7.86395511E-4, -0.00205520424, -8.4031286E-4,
                                7.32857734E-4, -1.49261235E-4, -0.00100950524};
  Vector<StorageT> expected_data{-0.00205520424, -1.49261235E-4};
  Vector<StorageT> pad_value{StorageT(0.000000e+00)};
  Vector<DimensionSize> edge_padding_low{0, -1};
  Vector<DimensionSize> edge_padding_high{0, -1};
  Vector<DimensionSize> interior_padding{0, 0};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32PadTest, kF32TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 0});

  Vector<StorageT> operand_data{-1.14107672E-4, -0.00149873865, 9.54493822E-4,
                                9.59243451E-4,  9.14431468E-4,  1.73450419E-4};
  Vector<StorageT> expected_data{};
  Vector<StorageT> pad_value{0.000000e+00};
  Vector<DimensionSize> edge_padding_low{0, -2};
  Vector<DimensionSize> edge_padding_high{0, -3};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

template <class T>
struct QuantizedIntPadTest : ::testing::Test {};
TYPED_TEST_SUITE(QuantizedIntPadTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedIntPadTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({2, 1});

  Vector<StorageT> operand_data{5, 8, 9, 6, 3, 1};
  Vector<StorageT> expected_data{8, 3};
  Vector<StorageT> pad_value{0};
  Vector<DimensionSize> edge_padding_low{0, -1};
  Vector<DimensionSize> edge_padding_high{0, -1};
  Vector<DimensionSize> interior_padding{0, 0};
  Vector<StorageT> output_data(shape_result.NumElements());

  const ExpressedT scale = static_cast<ExpressedT>(2);
  const StorageT zero_point = static_cast<StorageT>(0);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  Tensor operand{
      .type = QuantizedPerTensorTensorType{.shape = shape_operand,
                                           .element_type = tensor_type},
      .data = operand_data.data()};
  Tensor padding_value{
      .type = QuantizedPerTensorTensorType{.shape = shape_padding_value,
                                           .element_type = tensor_type},
      .data = pad_value.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_result,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
