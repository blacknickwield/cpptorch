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

template class Softmax<int>;
template class Softmax<float>;

}