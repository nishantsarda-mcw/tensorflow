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

using testing::ElementsAreArray;
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

TEST(StablehloDotGeneralModelTest, DotGeneralInt32_ScalarOutput) {
  TfLiteStablehloDotGeneralParams params = {
      {0},     // lhs_batching_dimensions;
      1,       // num_lhs_batching_dimensions
      {0},     // rhs_batching_dimensions;
      1,       // num_rhs_batching_dimensions
      {2, 1},  // lhs_contracting_dimensions;
      2,       // num_lhs_contracting_dimensions
      {1, 2},  // rhs_contracting_dimensions;
      2,       // num_rhs_contracting_dimensions
      2,       // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_INT32, {1, 3, 4}},
                                   {TensorType_INT32, {1, 4, 3}},
                                   {TensorType_INT32, {}}, params);

  model.SetInputs<int32_t>({2, 0, 0, 0, 5, -3, 0, 4, -1, 0, 0, -1},
                           {0, 4, 2, 3, 3, 3, -6, -2, 1, -1, 1, 0});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<int32_t> expected_values = {13};
  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray(expected_values));
}

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

TEST(StablehloDotGeneralModelTest, DotGeneralFloat32_EmptyBatchingDims) {
  TfLiteStablehloDotGeneralParams params = {
      {},   // lhs_batching_dimensions;
      0,    // num_lhs_batching_dimensions
      {},   // rhs_batching_dimensions;
      0,    // num_rhs_batching_dimensions
      {1},  // lhs_contracting_dimensions;
      1,    // num_lhs_contracting_dimensions
      {0},  // rhs_contracting_dimensions;
      1,    // num_rhs_contracting_dimensions
      2,    // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_FLOAT32, {4, 3}},
                                   {TensorType_FLOAT32, {3, 6}},
                                   {TensorType_FLOAT32, {}}, params);

  model.SetInputs<float>(
      {5.81311798, 2.08485532, 0.151162371, -1.21007407, -1.59476554,
       0.846119463, -0.83784312, -0.416278511, 1.24929118, 3.46354723,
       2.21915126, 3.81866336},
      {-2.10215521, -1.803730, -7.83739519, 4.36787844, 1.4788357, 3.10357666,
       -4.46420813, 0.879630148, -2.18081808, -1.95115197, -3.56435633,
       -0.671983778, -2.76886797, -0.212248296, 2.77085519, -1.21441388,
       -3.28464937, -4.60568237});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {
      -21.9458523, -8.6834774,  -49.6875458, 21.1395512,  0.668963671,
      15.9442625,  7.32033587,  0.600255728, 15.3061972,  -3.20136595,
      1.11560607,  -6.58085871, 0.160507768, 0.87991172,  10.9359407,
      -4.36453056, -3.85875082, -8.07441616, -27.7610416, -5.10577631,
      -21.4037914, 6.1610136,   -15.3307991, -8.329400};
  EXPECT_THAT(model.GetOutput<float>(),
              Pointwise(FloatNear(1e-5), expected_values));
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

TEST(StablehloDotGeneralModelTest, DotGeneralFloat16_MultipleBatchingDims) {
  TfLiteStablehloDotGeneralParams params = {
      {0, 3},  // lhs_batching_dimensions;
      2,       // num_lhs_batching_dimensions
      {0, 3},  // rhs_batching_dimensions;
      2,       // num_rhs_batching_dimensions
      {2},     // lhs_contracting_dimensions;
      1,       // num_lhs_contracting_dimensions
      {2},     // rhs_contracting_dimensions;
      1,       // num_rhs_contracting_dimensions
      2,       // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_FLOAT16, {2, 2, 2, 2}},
                                   {TensorType_FLOAT16, {2, 2, 2, 2}},
                                   {TensorType_FLOAT16, {}}, params);

  std::initializer_list<Eigen::half> lhs_data{
      Eigen::half(1.1),   Eigen::half(2.2),   Eigen::half(3.3),
      Eigen::half(4.3),   Eigen::half(5.5),   Eigen::half(6.0),
      Eigen::half(7.0),   Eigen::half(8.0),   Eigen::half(11.1),
      Eigen::half(12.22), Eigen::half(33.33), Eigen::half(44.32),
      Eigen::half(15.15), Eigen::half(6.6),   Eigen::half(7.3),
      Eigen::half(8.1)};
  std::initializer_list<Eigen::half> rhs_data{
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2),
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2),
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2),
      Eigen::half(1.2), Eigen::half(0.0), Eigen::half(0.0), Eigen::half(1.2)};
  model.SetInputs<Eigen::half>(lhs_data, rhs_data);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::half> expected_values{
      Eigen::half(1.319),  Eigen::half(1.319),  Eigen::half(6.6),
      Eigen::half(6.6),    Eigen::half(5.16),   Eigen::half(5.16),
      Eigen::half(9.6),    Eigen::half(9.6),    Eigen::half(13.32),
      Eigen::half(13.32),  Eigen::half(18.18),  Eigen::half(18.18),
      Eigen::half(53.184), Eigen::half(53.184), Eigen::half(9.7265),
      Eigen::half(9.7265)};
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

TEST(StablehloDotGeneralModelTest, DotGeneralBFloat16_MultipleContractingDims) {
  TfLiteStablehloDotGeneralParams params = {
      {0},     // lhs_batching_dimensions;
      1,       // num_lhs_batching_dimensions
      {0},     // rhs_batching_dimensions;
      1,       // num_rhs_batching_dimensions
      {2, 1},  // lhs_contracting_dimensions;
      2,       // num_lhs_contracting_dimensions
      {1, 2},  // rhs_contracting_dimensions;
      2,       // num_rhs_contracting_dimensions
      2,       // num_precision_configs
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT}  // precision config;
  };
  StablehloDotGeneralOpModel model({TensorType_BFLOAT16, {1, 3, 4}},
                                   {TensorType_BFLOAT16, {1, 4, 3}},
                                   {TensorType_BFLOAT16, {}}, params);

  std::initializer_list<Eigen::bfloat16> lhs_data{
      Eigen::bfloat16(4.968750e+00),  Eigen::bfloat16(-1.101560e+00),
      Eigen::bfloat16(1.015630e+00),  Eigen::bfloat16(4.812500e+00),
      Eigen::bfloat16(-3.398440e-01), Eigen::bfloat16(2.484380e+00),
      Eigen::bfloat16(-5.187500e+00), Eigen::bfloat16(-1.109380e+00),
      Eigen::bfloat16(-1.328130e+00), Eigen::bfloat16(3.312500e+00),
      Eigen::bfloat16(-4.937500e+00), Eigen::bfloat16(-4.281250e+00)};
  std::initializer_list<Eigen::bfloat16> rhs_data{
      Eigen::bfloat16(-1.164060e+00), Eigen::bfloat16(1.437500e+00),
      Eigen::bfloat16(-4.638670e-02), Eigen::bfloat16(-1.945310e+00),
      Eigen::bfloat16(-5.187500e+00), Eigen::bfloat16(-1.414060e+00),
      Eigen::bfloat16(-2.031250e+00), Eigen::bfloat16(-3.656250e+00),
      Eigen::bfloat16(-1.738280e-01), Eigen::bfloat16(4.902340e-01),
      Eigen::bfloat16(-5.968750e+00), Eigen::bfloat16(-3.671880e+00)};
  model.SetInputs<Eigen::bfloat16>(lhs_data, rhs_data);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::bfloat16> expected_values = {
      Eigen::bfloat16(2.087500e+01)};
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

}  // namespace
}  // namespace tflite
