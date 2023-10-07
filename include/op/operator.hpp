#ifndef CPPTORCH_INCLUDE_OPERATOR
#define CPPTORCH_INCLUDE_OPERATOR


namespace cpptorch {
template<class T>
class Tensor;

template<class T>
class Operator {
public:
    virtual ~Operator() = default;
    virtual auto operator()(const Tensor<T>&) -> Tensor<T> = 0;
};

}
#endif