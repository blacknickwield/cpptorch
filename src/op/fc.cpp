#include "op/fc.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cstddef>
// used for simd optimization
#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace cpptorch {

template<class T>
FullyConnected<T>::FullyConnected() {

}

template<class T>
FullyConnected<T>::FullyConnected(Tensor<T> &weights, Tensor<T> &bias) {
    
}

template<class T>
auto FullyConnected<T>::operator()(const Tensor<T> &tensor) -> Tensor<T> {
    // activation = wieghts * tensor + bias
    assert(tensor.size() == this->size());
    assert(tensor.m_shape.size() == this->m_weights.m_shape.size());
    size_t dims = this->m_weights.m_shape.size();
    for (size_t i = 0; i < dims; ++i) {
        assert(this->m_weights.m_shape[i] == tensor.m_shape[i]);
    }
// NEON SIMD optimization
// #ifdef __aarch64__

// #else

// #endif

    

}

}