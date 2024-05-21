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

#include "tensorflow/lite/experimental/shlo/tensor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/tensor_with_data.h"

using testing::Eq;
using testing::Pointwise;

namespace shlo_ref {
namespace {

TEST(TensorTest, BaselineTypeWorks) {
  EXPECT_EQ(BaselineType(DataType::kI1), DataType::kI1);
  EXPECT_EQ(BaselineType(DataType::kSI4), DataType::kSI4);
  EXPECT_EQ(BaselineType(DataType::kSI8), DataType::kSI8);
  EXPECT_EQ(BaselineType(DataType::kSI16), DataType::kSI16);
  EXPECT_EQ(BaselineType(DataType::kSI32), DataType::kSI32);
  EXPECT_EQ(BaselineType(DataType::kBF16), DataType::kBF16);
  EXPECT_EQ(BaselineType(DataType::kF16), DataType::kF16);
  EXPECT_EQ(BaselineType(DataType::kF32), DataType::kF32);
}

// Tensor::GetNdIndex
TEST(TensorTest, GetNdIndexWorks) {
  const Shape shape({1, 3, 3});
  std::vector<int32_t> data{5, 7, 9, 5, 4, 9, 7, 9, 8};

  Tensor tensor{
      .type = TensorType{.shape = shape, .element_type = DataType::kSI32},
      .data = data.data()};

  DimensionSize index = 5;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> indices(tensor.Rank());
  tensor.GetNdIndex(index, indices);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> expected_indices = {
      0, 1, 2};

  EXPECT_THAT(indices, Pointwise(Eq(), expected_indices));
}

// Tensor::FlattenIndex
TEST(TensorTest, FlattenIndexWorks) {
  const Shape shape({1, 3, 3});
  std::vector<int32_t> data{5, 7, 9, 5, 4, 9, 7, 9, 8};

  Tensor tensor{
      .type = TensorType{.shape = shape, .element_type = DataType::kSI32},
      .data = data.data()};

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> indices = {0, 1, 2};
  DimensionSize index_value = tensor.FlattenIndex(indices);
  DimensionSize expected_index_value = 5;

  EXPECT_THAT(expected_index_value, index_value);
}

// Tensor::Get and Tensor::Set
TEST(TensorTest, TensorGetSetWorks) {
  const Shape shape({1, 3, 3});
  std::vector<int32_t> data{3, 7, 9, 3, 4, 9, 7, 9, 8};

  Tensor tensor{
      .type = TensorType{.shape = shape, .element_type = DataType::kSI32},
      .data = data.data()};

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> indices = {0, 1, 2};
  tensor.Set<DataType::kSI32>(indices, 5);
  int32_t element = tensor.Get<DataType::kSI32>(indices);
  DimensionSize expected_element = 5;

  EXPECT_THAT(expected_element, element);
}

}  // namespace

}  // namespace shlo_ref
