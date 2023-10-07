#ifndef CPPTORCH_INCLUDE_MAXPOOLING
#define CPPTORCH_INCLUDE_MAXPOOLING

#include "op/operator.hpp"
#include "tensor.hpp"
#include <cstddef>

namespace cpptorch {

template<class T>
class MaxPooling :
    public Operator<T> {

public:
    MaxPooling();
    explicit MaxPooling(size_t stride_h, size_t stride_w, 
        size_t padding_h, size_t padding_w);
public:
    auto operator()(const Tensor<T> &tensor) -> Tensor<T> override;
private:
    size_t m_stride_h;
    size_t m_stride_w;
    size_t m_padding_h;
    size_t m_padding_w;

};


}

#endif