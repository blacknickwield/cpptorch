#include <cassert>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include "tensor.hpp"

using namespace cpptorch;
TEST(TensorTest, SliceTest) {
    const std::vector<size_t> tensor_shapes = {5, 6, 7};
    Tensor<int> tensor(tensor_shapes);
    const std::vector<size_t> slice_shapes = {1, 2};
    auto slice = tensor({1, 2});
    ASSERT_EQ(tensor.shapes(), tensor_shapes);
    
    auto shapes = slice.shapes();
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(shapes, std::vector<size_t>({7}));
}

TEST(TensorTest, ValueTest) {
    const size_t size = 15;
    const std::vector<size_t> shapes = {1, 3, 5};
    int *data = new int[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = i;
    }

    Tensor<int> tensor(data, shapes);
    auto x = tensor.shapes();

    ASSERT_EQ((tensor[{0, 2, 3}]), 13);
}

TEST(TensorTest, RangeSliceTest) {
    const std::vector<size_t> shape = {8, 9, 10};
    const size_t size = 720;
    int *data = new int[size];
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 9; ++j) {
            for (size_t k = 0; k < 10; ++k) {
                size_t index = i * 90 + j * 10 + k;
                data[index] = index;
            }
        }
    }
    Tensor<int> tensor(data, shape);
    const std::vector<Tensor<int>::Shape> indexs = {{1, 2}, {3, 5}, {5, 8}};
    Tensor<int> slice = tensor.slice(indexs);
    std::vector<size_t> slice_shape = slice.shapes();

    ASSERT_EQ(slice_shape.size(), shape.size());
    ASSERT_GE(slice_shape.size(), indexs.size());
    for (size_t i = 0; i < indexs.size(); ++i) {
        ASSERT_EQ(slice_shape[i], (indexs[i].second - indexs[i].first));
    }

    for (size_t i = 0; i < 1; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                size_t value = (1 + i) * 90 + (3 + j) * 10 + (5 + k);
                std::cout << slice[{i, j, k}] << ' ' << value << std::endl;
                ASSERT_EQ((slice[{i, j, k}]), value);
            }
        }
    }
}