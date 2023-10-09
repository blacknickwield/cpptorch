#ifndef CPPTORCH_INCLUDE_TENSOR
#define CPPTORCH_INCLUDE_TENSOR
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include "storage.hpp"
namespace cpptorch {

template<class T>
class Tensor {
public:
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape, bool requires_grad = false);
    explicit Tensor(T *data, const std::vector<size_t> &shape, bool requires_grad = false);
    // Tensor()
    // copy constructor
    // Tensor(const Tensor<T>&) = delete;
    // auto operator=(const Tensor<T>&) -> Tensor<T>& = delete;
    auto operator=(const Tensor<T>&) -> Tensor<T>&;
    Tensor(const Tensor<T>&);
    

    Tensor(Tensor<T> &&);
    auto operator=(Tensor<T> &&) -> Tensor<T>&;
    // move constructor

    
    using SliceIdx = std::vector<std::pair<size_t, size_t>>;
private:
    explicit Tensor(const std::vector<size_t> &shape, std::shared_ptr<Storage<T>> storage);
public:
    static auto clone(const Tensor<T>&) -> Tensor<T>;
public:
    auto size() const -> size_t;
    auto shapes() const -> const std::vector<size_t>&;
public:
    // auto fill(T value) -> void;
    /*
    * _ indicates that the method is a inplace one.
    * That is to say, it will change elements that m_storage holds.
    * So the return value is void. By contrast, no _ indicates that
    * it will produce a new Tensor. So the return value is Tensor<T>
    */
    auto fill(T value) const -> Tensor<T>;
    auto fill_(T value) -> void;
    auto reshape(const std::vector<size_t> &shape) -> void;
    auto view(const std::vector<size_t> &shape) -> Tensor<T>;
    auto element_wise(const std::function<T(T)>&) const -> Tensor<T>;
    auto element_wise_(const std::function<T(T)>&) -> void;
    auto slice(SliceIdx &) -> Tensor<T>;
    auto backward() -> void;
public:
    auto dot(const Tensor<T> &tensor) const -> Tensor<T>;
    auto dot_(const Tensor<T> &tensor) -> Tensor<T>&;
public:
    // auto operator=(const Tensor<T> &tensor) -> Tensor<T>&;
    auto operator+(const Tensor<T> &tensor) const -> Tensor<T>;
    auto operator+=(const Tensor<T> &tensor) -> Tensor<T>&;
    auto operator-(const Tensor<T> &tensor) const -> Tensor<T>;
    auto operator-=(const Tensor<T> &tensor) -> Tensor<T>&;
    auto operator*(const T scale) const -> Tensor<T>;
    auto operator*=(const T scale) -> Tensor<T>&;
    auto operator*(const Tensor<T> &tensor) const -> Tensor<T>;
    auto operator*=(const Tensor<T> &tensor) -> Tensor<T>&;
    auto operator/(const T scale) const -> Tensor<T>;
    auto operator/=(const T scale) -> Tensor<T>&;
    // auto operator[](size_t index) const -> T&;
    // auto operator()(size_t index, ...) const -> Tensor<T>;
    auto operator()(const std::vector<size_t> &indexs) const -> Tensor<T>;

    auto operator()(std::initializer_list<size_t>) const -> Tensor<T>;

public:
    auto operator[](std::initializer_list<size_t> indexs) -> T&;
    auto operator[](std::initializer_list<size_t> indexs) const -> const T&;
private:
    auto operator[](size_t index) -> T&;
    auto operator[](size_t index) const -> const T&;
   
private:
    std::vector<size_t> m_shape;
    std::shared_ptr<Storage<T>> m_storage;
    bool m_requires_grad;
    std::unique_ptr<Tensor<T>> m_gradient;

    std::function<size_t(size_t)> m_index_cb;
};


}
#endif