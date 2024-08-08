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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::FloatEq;
using testing::FloatNear;
using testing::Pointwise;

class StablehloDotGeneralOpModel : public SingleOpModel {
 public:
  StablehloDotGeneralOpModel(const TensorData& lhs, const TensorData& rhs,
                             const TensorData& output,
                             const TfLiteStablehloDotGeneralParams& params) {
    lhs_ = AddInput(lhs);
    rhs_ = AddInput(rhs);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_DOT_GENERAL,
        BuiltinOptions2_StablehloDotGeneralOptions,
        CreateStablehloDotGeneralOptions(
            builder_,
            builder_.CreateVector(
                std::vector(params.lhs_batching_dimensions,
                            params.lhs_batching_dimensions +
                                params.num_lhs_batching_dimensions)),
            builder_.CreateVector(
                std::vector(params.rhs_batching_dimensions,
                            params.rhs_batching_dimensions +
                                params.num_rhs_batching_dimensions)),
            builder_.CreateVector(
                std::vector(params.lhs_contracting_dimensions,
                            params.lhs_contracting_dimensions +
                                params.num_lhs_contracting_dimensions)),
            builder_.CreateVector(
                std::vector(params.rhs_contracting_dimensions,
                            params.rhs_contracting_dimensions +
                                params.num_rhs_contracting_dimensions)),
            builder_.CreateVector(std::vector(
                params.precision_config,
                params.precision_config + params.num_precision_configs)))
            .Union());
    BuildInterpreter({GetShape(lhs_), GetShape(rhs_)});
  }

  template <typename T>
  void SetInputs(std::initializer_list<T> data_lhs,
                 std::initializer_list<T> data_rhs) {
    PopulateTensor<T>(lhs_, data_lhs);
    PopulateTensor<T>(rhs_, data_rhs);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int lhs_;
  int rhs_;
  int output_;
};

TEST(StablehloDotGeneralModelTest, DotGeneralFloat32) {
  TfLiteStablehloDotGeneralParams params = {
      {0},  // lhs_batching_dimensions;
      1,    // num_lhs_batching_dimensions
      {0},  // rhs_batching_dimensions;
      1,    // num_rhs_batching_dimensions
      {2},  // lhs_contracting_dimensions;
      1,    // num_lhs_contracting_dimensions
      {2},  // rhs_contracting_dimensions;
      1,    // num_rhs_contracting_dimensions
      2,    // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_FLOAT32, {2, 2, 2}},
                                   {TensorType_FLOAT32, {2, 2, 2}},
                                   {TensorType_FLOAT32, {}}, params);

  model.SetInputs<float>({1.1, 2.2, 3.3, 4.3, 5.5, 6.0, 7.0, 8.0},
                         {1.2, 0.0, 0.0, 1.2, 1.2, 0.0, 0.0, 1.2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1.32, 2.64, 3.96, 5.16,
                                        6.6,  7.2,  8.4,  9.6};
  EXPECT_THAT(model.GetOutput<float>(), Pointwise(FloatEq(), expected_values));
}

TEST(StablehloDotGeneralModelTest, DotGeneralFloat16) {
  TfLiteStablehloDotGeneralParams params = {
      {0},  // lhs_batching_dimensions;
      1,    // num_lhs_batching_dimensions
      {0},  // rhs_batching_dimensions;
      1,    // num_rhs_batching_dimensions
      {2},  // lhs_contracting_dimensions;
      1,    // num_lhs_contracting_dimensions
      {2},  // rhs_contracting_dimensions;
      1,    // num_rhs_contracting_dimensions
      2,    // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_FLOAT16, {2, 2, 2}},
                                   {TensorType_FLOAT16, {2, 2, 2}},
                                   {TensorType_FLOAT16, {}}, params);

  std::initializer_list<Eigen::half> lhs_data{
      Eigen::half(1.1), Eigen::half(2.2), Eigen::half(3.3), Eigen::half(4.3),
      Eigen::half(5.5), Eigen::half(6.0), Eigen::half(7.0), Eigen::half(8.0)};
  std::initializer_list<Eigen::half> rhs_data{
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2),
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2)};
  model.SetInputs<Eigen::half>(lhs_data, rhs_data);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::half> expected_values{
      Eigen::half(1.31934), Eigen::half(2.6386), Eigen::half(3.9609),
      Eigen::half(5.1601),  Eigen::half(6.6015), Eigen::half(7.2031),
      Eigen::half(8.3984),  Eigen::half(9.6015)};
  EXPECT_THAT(model.GetOutput<Eigen::half>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

TEST(StablehloDotGeneralModelTest, DotGeneralBFloat16) {
  TfLiteStablehloDotGeneralParams params = {
      {0},  // lhs_batching_dimensions;
      1,    // num_lhs_batching_dimensions
      {0},  // rhs_batching_dimensions;
      1,    // num_rhs_batching_dimensions
      {2},  // lhs_contracting_dimensions;
      1,    // num_lhs_contracting_dimensions
      {2},  // rhs_contracting_dimensions;
      1,    // num_rhs_contracting_dimensions
      2,    // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_BFLOAT16, {2, 2, 2}},
                                   {TensorType_BFLOAT16, {2, 2, 2}},
                                   {TensorType_BFLOAT16, {}}, params);

  std::initializer_list<Eigen::bfloat16> lhs_data{
      Eigen::bfloat16(1.1), Eigen::bfloat16(2.2), Eigen::bfloat16(3.3),
      Eigen::bfloat16(4.3), Eigen::bfloat16(5.5), Eigen::bfloat16(6.0),
      Eigen::bfloat16(7.0), Eigen::bfloat16(8.0)};
  std::initializer_list<Eigen::bfloat16> rhs_data{
      Eigen::bfloat16(1.2), Eigen::bfloat16(0.0), Eigen::bfloat16(0.0),
      Eigen::bfloat16(1.2), Eigen::bfloat16(1.2), Eigen::bfloat16(0.0),
      Eigen::bfloat16(0.0), Eigen::bfloat16(1.2)};
  model.SetInputs<Eigen::bfloat16>(lhs_data, rhs_data);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::bfloat16> expected_values = {
      Eigen::bfloat16(1.32813), Eigen::bfloat16(2.6562),
      Eigen::bfloat16(3.96875), Eigen::bfloat16(5.1875),
      Eigen::bfloat16(6.6250),  Eigen::bfloat16(7.21875),
      Eigen::bfloat16(8.4375),  Eigen::bfloat16(9.6250)};
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

}  // namespace
}  // namespace tflite
