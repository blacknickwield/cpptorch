#ifndef CPPTORCH_INCLUDE_RELU
#define CPPTORCH_INCLUDE_RELU

#include "operator.hpp"
#include "tensor.hpp"
#include <memory>
namespace cpptorch {
template<class T>
class ReLU :
    public Operator<T> {
public:
    ReLU();
public:
    auto operator()(const Tensor<T>&) -> Tensor<T> override; 
    auto backward(const Tensor<T> &gradient_output) -> Tensor<T> override;
    // std::weak_ptr<Tensor<T>> input;
private:
    Tensor<T> input;
};

}
#endif