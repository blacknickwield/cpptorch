#include "op/conv.hpp"
#include "tensor.hpp"
#include <cstddef>

namespace cpptorch {

template<class T>
Conv<T>::Conv() {

}

template<class T>
Conv<T>::Conv(size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {

}

template<class T>
auto Conv<T>::operator()(const Tensor<T> &tensor) -> Tensor<T> {
    // TODO(filtering)

    
}
}