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
    
    Tensor<T> activations = tensor.element_wise(relu);
    return activations;
}

template class ReLU<int>;
template class ReLU<float>;
}