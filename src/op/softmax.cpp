#include "op/softmax.hpp"
#include "tensor.hpp"

#include <cmath>

namespace cpptorch {
template<class T>
Softmax<T>::Softmax() {

}

template<class T>
auto Softmax<T>::operator()(const Tensor<T> &tensor) -> Tensor<T> {
    // as element-wise funciton
    auto softmax = [] (const T& value) -> T {
        const T value_exp = std::exp(value);
        return value_exp / (1 - value_exp);
    };

    Tensor<T> activations = tensor.element_wise(softmax);
    return activations;
}

template<class T>
auto Softmax<T>::backward(const Tensor<T> &gradient_output) -> Tensor<T> {
    const auto softmax_grad_fn = [] (const T &value) -> T {
        const T value_exp = std::exp(value);
        return 1. / std::pow(1 - value_exp, 2);
    };

    Tensor<T> softmax_gradient = this->input.element_wise(softmax_grad_fn);
    Tensor<T> gradient = gradient_output.dot(softmax_gradient);

    return gradient;
}

template class Softmax<int>;
template class Softmax<float>;

}