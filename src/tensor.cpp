#include "tensor.hpp"
#include "storage.hpp"
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

namespace cpptorch {
template<class T>
Tensor<T>::Tensor() {

}

template<class T>
Tensor<T>::Tensor(const std::vector<size_t> &dims, bool requires_grad) 
    : m_dims(dims), m_shape(dims.size()), m_requires_grad(requires_grad), m_gradient(nullptr) {
    std::transform(dims.begin(), dims.end(), this->m_shape.begin(), [] (const size_t &index) -> Shape {
        return {0, index};
    });

    auto size = std::reduce(m_shape.begin(), m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });

    this->m_storage = std::make_shared<Storage<T>>(size);
}

template<class T>
Tensor<T>::Tensor(const std::vector<Shape> &shape, bool requires_grad) 
    : m_shape(shape), m_requires_grad(requires_grad), m_gradient(nullptr) {
    size_t size = std::reduce(this->m_shape.begin(), this->m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });

    this->m_storage = std::make_shared<Storage<T>>(size);
}

template<class T>
Tensor<T>::Tensor(T *data, const std::vector<size_t> &dims, bool requires_grad) 
    : m_dims(dims), m_shape(dims.size()), m_requires_grad(requires_grad), m_gradient(nullptr) {
    std::transform(dims.begin(), dims.end(), this->m_shape.begin(), [] (const size_t &index) -> Shape {
        return {0, index};
    });
    
    size_t size = std::reduce(dims.begin(), dims.end(), 1, [] (const size_t &a, const size_t &b) -> size_t {
        return a * b;
    });
    
    this->m_storage = std::make_shared<Storage<T>>(data, size);
}

template<class T>
Tensor<T>::Tensor(T *data, const std::vector<Shape> &shape, bool requires_grad) :
    m_shape(shape), m_requires_grad(requires_grad), m_gradient(nullptr) {

    size_t size = 1;
    for (const auto &[start, end] : this->m_shape) {
        size *= (end - start);
    }
    // TODO
}

template<class T>
Tensor<T>::Tensor(const std::vector<Shape> &shape, std::shared_ptr<Storage<T>> storage) :
    m_shape(shape.size()), m_storage(storage) {
    std::transform(shape.begin(), shape.end(), this->m_shape.begin(), [] (const Shape &shape) -> Shape {
        return {START(shape), END(shape)};
    });
}

template<class T>
auto Tensor<T>::operator=(const Tensor<T> &tensor) -> Tensor<T>& {
    this->m_shape = tensor.m_shape;
    this->m_requires_grad = tensor.m_requires_grad;
    this->m_storage = tensor.m_storage;
    this->m_index_cb = tensor.m_index_cb;

    if (this->m_requires_grad) {
        //TODO: gradient copy
    }
    
    return *this;
}

template<class T>
Tensor<T>::Tensor(const Tensor<T> &tensor) 
    : m_shape(tensor.m_shape), m_requires_grad(tensor.m_requires_grad), m_storage(tensor.m_storage), m_index_cb(tensor.m_index_cb) {
    if (this->m_requires_grad) {
        //TODO: gradient copy
    }
}



template<class T>
Tensor<T>::Tensor(Tensor<T> &&tensor) 
    : m_shape(std::move(tensor.m_shape)), m_requires_grad(std::move(tensor.m_requires_grad)),
    m_gradient(std::move(tensor.m_gradient)), m_storage(tensor.m_storage) {

}

template<class T>
auto Tensor<T>::operator=(Tensor<T> &&tensor) -> Tensor<T>& {
    if (this != &tensor) {
        this->m_shape = std::move(tensor.m_shape);
        this->m_requires_grad = std::move(tensor.m_requires_grad);
        this->m_gradient = std::move(tensor.m_gradient);
        this->m_storage = tensor.m_storage;
    }
    return *this;
}


template<class T>
auto Tensor<T>::size() const -> size_t {
    return std::reduce(this->m_shape.begin(), this->m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });
}

// template<class T>
// auto Tensor<T>::shapes() const -> const std::vector<size_t>& {
//     return this->m_shape;
// }
template<class T>
auto Tensor<T>::shapes() const -> std::vector<size_t> {
    std::vector<size_t> shape(this->m_shape.size());
    std::transform(this->m_shape.begin(), this->m_shape.end(), shape.begin(), [] (const Shape &rg) {
        return END(rg) - START(rg);
    });
    return shape;
}

template<class T>
auto Tensor<T>::dims() const -> const std::vector<size_t>& {
    return this->m_dims;
}

template<class T>
auto Tensor<T>::fill(T value) const -> Tensor<T> {
    // this->m_storage->fill(value);
    Tensor<T> tensor(this->m_shape);
    tensor.fill_(value);
    return tensor;
}

template<class T>
auto Tensor<T>::fill_(T value) -> void {
    this->m_storage->fill(value);
}

template<class T>
auto Tensor<T>::view(const std::vector<size_t> &shape) -> Tensor<T> {
    auto tensor = Tensor<T>(shape);
    return tensor;
}


template<class T>
auto Tensor<T>::element_wise(const std::function<T(T)> &trans) const -> Tensor<T> {
    Tensor<T> tensor(this->m_shape, this->m_storage);
    tensor.element_wise_(trans);
    return tensor;
    // this->m_storage->element_wise(trans);

}

template<class T>
auto Tensor<T>::element_wise_(const std::function<T(T)> &trans) -> void {
    this->m_storage->element_wise(trans);
}


template<class T>
auto Tensor<T>::slice(const std::vector<Shape> &indexs) -> Tensor<T> {
    assert(indexs.size() <= this->m_shape.size());
    std::vector<Shape> shape;
    for (const auto &[start, end] : indexs) {
        assert(start <= end);
        shape.push_back({start, end});
    }

    
    assert(shape.size() <= this->m_shape.size());
    Tensor<T> tensor(shape);
    tensor.m_dims = this->m_dims;
    tensor.m_storage = this->m_storage;

    size_t stride = std::reduce(this->m_dims.begin(), this->m_dims.end(), 1, [] (const size_t &a, const size_t &b) -> size_t {
        return a * b;
    });

    size_t index = 0;
    size_t m_shape_index = 0;

    for (const auto &[start, end] : indexs) {
        stride /= this->m_dims[m_shape_index];
        index += start * stride;
        ++m_shape_index;
    }


    // tensor.m_index_cb = [base=index, &shape=tensor.m_shape] (size_t index) -> size_t {
    //     return base + index;
    // };

    return tensor;
}

template<class T>
auto Tensor<T>::backward() -> void {
    // backward gradient according to computing graph
    

}

// template<class T>
// auto Tensor<T>::operator=(const Tensor<T> &tensor) -> Tensor<T>& {
//     this->m_shape = tensor.m_shape;
//     this->m_storage = tensor.m_storage;
//     return *this;
// }


template<class T>
auto Tensor<T>::dot(const Tensor<T> &tensor) const -> Tensor<T> {
    assert(this->size() == tensor.size());
    assert(this->m_shape.size() == tensor.size());
    for (size_t i = 0; i < this->m_shape.size(); ++i) {
        assert(this->m_shape[i] == tensor.m_shape[i]);
    }

    T *data = new T[tensor.size()];
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = tensor[i] * (*this)[i];
    }

    Tensor<T> result(data, this->m_shape);
    return result;
}

template<class T>
auto Tensor<T>::dot_(const Tensor<T> &tensor) -> Tensor<T>& {
    assert(this->size() == tensor.size());
    assert(this->m_shape.size() == tensor.shapes().size());
    for (size_t i = 0; i < this->m_shape.size(); ++i) {
        assert(this->m_shape[i] == tensor.m_shape[i]);
    }

    for (size_t i = 0; i < tensor.size(); ++i) {
        (*this)[i] *= tensor[i];
    }

    return *this;
}


template<class T>
auto Tensor<T>::operator+(const Tensor<T> &tensor) const -> Tensor<T> {
    assert(tensor.size() == this->size());
    size_t size = this->size();
    T *data = new T[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = tensor[i] + (*this)[i];
    }

    Tensor<T> t(data, this->m_shape);
    return t;
}


template<class T>
auto Tensor<T>::operator-(const Tensor<T> &tensor) const -> Tensor<T> {
    assert(tensor.size() == this->size());
    size_t size = this->size();
    T *data = new T[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = tensor[i] - (*this)[i];
    }

    Tensor<T> t(data, this->m_shape);
    return t;
}

template<class T>
auto Tensor<T>::operator*(const T scale) const -> Tensor<T> {
    size_t size = this->size();
    T *data = new T[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = (*this)[i] * scale;
    }

    Tensor<T> t(data, this->m_shape);
    return t;

}

template<class T>
auto Tensor<T>::operator/(const T scale) const -> Tensor<T> {
    size_t size = this->size();
    T *data = new T[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = (*this)[i] / scale;
    }

    Tensor<T> t(data, this->m_shape);
    return t;
}

template<class T>
auto Tensor<T>::operator+=(const Tensor<T> &tensor) -> Tensor<T>& {
    assert(this->size() == tensor.size());
    size_t size = this->size();
    for (int i = 0; i < size; ++i) {
        (*this)[i] += tensor[i];
    }
    return *this;
}

template<class T>
auto Tensor<T>::operator-=(const Tensor<T> &tensor) -> Tensor<T>& {
    assert(this->size() == tensor.size());
    size_t size = this->size();
    for (int i = 0; i < size; ++i) {
        (*this)[i] -= tensor[i];
    }

    return *this;


}

template<class T>
auto Tensor<T>::operator*=(const T scale) -> Tensor<T>& {
    const auto trans = [&scale] (const T &value) -> T {
        return scale * value;
    };
    
    this->m_storage->element_wise(trans);
    return *this;
}


template<class T>
auto Tensor<T>::operator/=(const T scale) -> Tensor<T>& {
    const auto trans = [&scale] (const T &value) -> T {
        return value / scale;
    };
    this->m_storage->element_wise(trans);
    return *this;
}


// template<class T>
// auto Tensor<T>::operator()(size_t index, ...) const -> Tensor<T> {
//     va_list args;
//     va_start(args, index);
// }

template<class T>
auto Tensor<T>::operator()(const std::vector<size_t> &indexs) const -> Tensor<T> {
    assert(indexs.size() <= this->m_shape.size()); 
    
    // std::vector<size_t> shape;
    std::vector<Shape> shape;
    for (size_t i = indexs.size(); i < this->m_shape.size(); ++i) {
        shape.push_back({START(this->m_shape[i]), END(this->m_shape[i])});
    }
    Tensor<T> tensor(shape);
    tensor.m_storage = this->m_storage;

    // size_t s_index = 0;
    // size_t volumn = 1;
    // for (auto index_it = index.rbegin(), shape_it = this->m_shape.rbegin(); index_it != index.rend() && shape_it != m_shape.rend(); ++index_it, ++shape_it) {
    //     s_index += *index_it * volumn;
    //     volumn *= *shape_it;
    // }
    
    size_t stride = std::reduce(this->m_shape.begin(), this->m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });

    size_t index = 0;
    size_t m_shape_index = 0;

    for (const auto &in : indexs) {
        size_t rg = END(this->m_shape[m_shape_index]) - START(this->m_shape[m_shape_index]);
        assert(in < rg);
        stride /= rg;
        index += in * stride;
        ++m_shape_index;
    }

    tensor.m_index_cb = [base = index] (size_t index) -> size_t {
        return base + index;
    };
 
    return tensor;
}

template<class T>
auto Tensor<T>::operator()(std::initializer_list<size_t> indexs) const -> Tensor<T> {
    assert(indexs.size() <= this->m_shape.size());
    // std::vector<size_t> shape;
    std::vector<Shape> shape;
    for (size_t i = indexs.size(); i < this->m_shape.size(); ++i) {
        shape.push_back(this->m_shape[i]);
    }
    Tensor<T> tensor(shape);
    tensor.m_storage = this->m_storage;

    size_t stride = std::reduce(this->m_shape.begin(), this->m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });

    size_t index = 0;
    size_t m_shape_index = 0;

    for (const auto &in : indexs) {
        size_t rg = END(this->m_shape[m_shape_index]) - START(this->m_shape[m_shape_index]);
        assert(in < rg);
        stride /= rg;
        index += in * stride;
        ++m_shape_index;
    }

    tensor.m_index_cb = [base=index] (const size_t &index) -> size_t {
        return base + index;
    };

    return tensor;
}

template<class T>
auto Tensor<T>::operator[](std::initializer_list<size_t> indexs) -> T& {
    assert(indexs.size() == this->m_shape.size());
    
    size_t stride = std::reduce(this->m_dims.begin(), this->m_dims.end(), 1, [] (const size_t &a, const size_t &b) -> size_t {
        return a * b;
    });

    size_t index = 0;
    size_t m_shape_index = 0;

    for (const auto &in : indexs) {
        stride /= this->m_dims[m_shape_index];
        index += (in + START(this->m_shape[m_shape_index])) * stride;
        ++m_shape_index;
    }

    if (this->m_index_cb) {
        index = this->m_index_cb(index);
    }

    return (*this->m_storage)[index];
}

template<class T>
auto Tensor<T>::operator[](std::initializer_list<size_t> indexs) const -> const T& {
    assert(indexs.size() == this->m_shape.size());
    
    size_t stride = std::reduce(this->m_shape.begin(), this->m_shape.end(), 1, [] (const size_t &a, const Shape &b) -> size_t {
        return a * (END(b) - START(b));
    });


    size_t index = 0;
    size_t m_shape_index = 0;
    for (const auto &in : indexs) {
        // assert(this->m_shape[m_shape_index] > in);
        // stride /= this->m_shape[m_shape_index];
        size_t rg = END(this->m_shape[m_shape_index]) - START(this->m_shape[m_shape_index]);
        assert(rg > in);
        stride /= rg;
        
        index += in * stride;
        ++m_shape_index;
    }

    if (this->m_index_cb) {
        index = this->m_index_cb(index);
    }
    
    return (*this->m_storage)[index];
}

template<class T>
auto Tensor<T>::operator[](size_t index) -> T& {
    assert(index < this->size());
    if (this->m_index_cb) {
        index = this->m_index_cb(index);
    }
    return (*m_storage)[index];
}

template<class T>
auto Tensor<T>::operator[](size_t index) const -> const T& {
    assert(index < this->size());
    if (this->m_index_cb) {
        index = this->m_index_cb(index);
    }
    return (*m_storage)[index];
}


template<class T>
auto Tensor<T>::clone(const Tensor<T> &tensor) -> Tensor<T> {
    Tensor<T> copy = Tensor<T>(tensor.m_shape, tensor.m_requires_grad);
    copy.m_storage = tensor.m_storage;
    if (tensor.m_requires_grad == true) {
        // TODO: gradient copy
        // copy.m_gradient = Tensor<T>::clone(*tensor.m_gradient.get());
    }

    return copy;
}

template class Tensor<int>;
template class Tensor<float>;

}