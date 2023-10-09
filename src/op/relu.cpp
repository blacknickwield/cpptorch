#include "op/relu.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
namespace cpptorch {

template<class T>
ReLU<T>::ReLU() {

}

template<class T>
auto ReLU<T>::operator()(const Tensor<T> &tensor) -> Tensor<T> {
    const auto relu = [] (const T& value) -> T {
        if (value < 0) {
            return 0;
        }
        return value;
    };
    this->input = tensor;
    Tensor<T> activations = tensor.element_wise(relu);
    return activations;
}

template<class T>
auto ReLU<T>::backward(const Tensor<T> &gradient_output) -> Tensor<T> {
    const auto relu_grad_fn = [] (const T &value) -> T {
        return value < 0 ? 0 : 1;
    };
    
    Tensor<T> relu_gradient = this->input.element_wise(relu_grad_fn);
    Tensor<T> gradient = gradient_output.dot(relu_gradient);
    return gradient;
}

template class ReLU<int>;
template class ReLU<float>;
}