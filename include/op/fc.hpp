#ifndef CPPTORCH_INCLUDE_FC
#define CPPTORCH_INCLUDE_FC

#include "operator.hpp"
#include "tensor.hpp"

namespace cpptorch {

template<class T>
class FullyConnected :
    public Operator<T> {
public:
    FullyConnected();
    explicit FullyConnected(Tensor<T> &weights, Tensor<T> &bias);
public:
    auto operator()(const Tensor<T> &tensor) -> Tensor<T> override;

private:
    Tensor<T> m_weights;
    Tensor<T> m_bias;
};

}

#endif