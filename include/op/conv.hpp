#ifndef CPPTORCH_INCLUDE_CONV
#define CPPTORCH_INCLUDE_CONV

#include "operator.hpp"
#include "tensor.hpp"
#include <cstddef>

namespace cpptorch {

template<class T>
class Conv :
    public Operator<T> {
public:
    Conv();
    explicit Conv(size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w);
    // explicit Conv()

public:
    auto operator()(const Tensor<T> &tensor) -> Tensor<T> override;
private:
    Tensor<T> m_kernel;
    size_t m_stride_h;
    size_t m_stride_w;
    size_t m_padding_h;
    size_t m_padding_w;
};


}
#endif