#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include "storage.hpp"

TEST(StorageTest, FillTest) {
    const size_t size = 10;
    const int value = 114514;
    auto storage = std::make_shared<cpptorch::Storage<int>>(size);
    storage->fill(value);
    for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(value, (*storage)[i]);
    }
}

TEST(StorageTest, CopyTest) {
    const size_t size = 10;
    const int value = 1919810;
    cpptorch::Storage<int> storage(size);
    storage.fill(value);

    auto storage_copy = storage;
    ASSERT_EQ(storage.size(), storage_copy.size());
    for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(storage[i], storage_copy[i]);
    }
}

TEST(StorageTest, ElementWiseTest) {
    const size_t size = 10;
    cpptorch::Storage<int> storage(size);
    for (size_t i = 0; i < size; ++i) {
        storage[i] = i;
    }

    const int scale = 2;
    auto trans = [&scale] (const int value) -> int {
        return value * scale;
    };
    storage.element_wise(trans);
    for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(storage[i], i * scale);
    }
}