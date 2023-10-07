#ifndef CPPTORCH_INCLUDE_RELU
#define CPPTORCH_INCLUDE_RELU

#include "operator.hpp"
#include "tensor.hpp"
namespace cpptorch {
template<class T>
class ReLU :
    public Operator<T> {
public:
    ReLU();
public:
    auto operator()(const Tensor<T>&) -> Tensor<T> override; 
};

}
#endif