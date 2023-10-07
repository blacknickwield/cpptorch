#include "op/max_pooling.hpp"
#include "tensor.hpp"
#include <cstddef>

namespace cpptorch {

template<class T>
MaxPooling<T>::MaxPooling() {

}

template<class T>
MaxPooling<T>::MaxPooling(size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) : 
    m_stride_h(stride_h), m_stride_w(stride_w), m_padding_h(padding_h), m_padding_w(padding_w) {

}

template<class T>
auto MaxPooling<T>::operator()(const Tensor<T> &tensor) -> Tensor<T> {
    
}


}