#ifndef CPPTORCH_INCLUDE_SOFTMAX
#define CPPTORCH_INCLUDE_SOFTMAX

#include "op/conv.hpp"
#include "op/operator.hpp"
#include "tensor.hpp"
namespace cpptorch {

template<class T>
class Softmax :
    public Operator<T> {
public:
    Softmax();
public:
    auto operator()(const Tensor<T> &tensor) -> Tensor<T> override;
    auto backward(const Tensor<T> &gradient_output) -> Tensor<T> override;
private:
    Tensor<T> input;
};

}
#endif