cmake_minimum_required(VERSION 3.25)

project(cpptorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)
file(GLOB_RECURSE SRC
    src/*.cpp
    src/op/*.cpp)

add_executable(${PROJECT_NAME} ${SRC} main.cpp)