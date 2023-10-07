#include <cassert>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
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
    
    ASSERT_EQ((tensor[{0, 2, 3}]), 13);
}