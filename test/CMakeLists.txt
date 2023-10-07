cmake_minimum_required(VERSION 3.22)

project(cpptorch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)

if(GTest_FOUND)  
  message(STATUS "Google Test found: ${GTest_INCLUDE_DIRS}, ${GTest_LIBRARIES}")  
else()  
  message(FATAL_ERROR "Google Test not found!")  
endif()

enable_testing()

add_executable(StorageTest StorageTest.cpp ../src/storage.cpp)  
target_link_libraries(StorageTest GTest::gtest_main)
target_include_directories(StorageTest PUBLIC ../include)

add_executable(TensorTest TensorTest.cpp ../src/tensor.cpp ../src/storage.cpp)  
target_link_libraries(TensorTest GTest::gtest_main)
target_include_directories(TensorTest PUBLIC ../include)

add_executable(OpTest OpTest.cpp ../src/tensor.cpp ../src/storage.cpp ../src/op/relu.cpp ../src/op/softmax.cpp)  
target_link_libraries(OpTest GTest::gtest_main)
target_include_directories(OpTest PUBLIC ../include)

include(GoogleTest)
gtest_add_tests(TARGET StorageTest)