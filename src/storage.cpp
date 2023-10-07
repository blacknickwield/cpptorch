#include "storage.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <iostream>

namespace cpptorch {
/*
Storage is a 1-D memory manager.
It can only be accessed through 1 index.
*/
template <class T> Storage<T>::Storage() : m_size(0), m_data(nullptr) {}

template <class T>
Storage<T>::Storage(size_t n) : m_size(n), m_data(std::make_unique<T[]>(n)) {}

template<class T>
Storage<T>::Storage(T *data, size_t size) : m_size(size) {
  this->m_data = std::unique_ptr<T[]>(data);
}

template<class T>
Storage<T>::Storage(const Storage<T>& storage) : m_size(storage.m_size) {
  this->m_data = std::make_unique<T[]>(this->m_size);
  std::transform(storage.m_data.get(), storage.m_data.get() + m_size, this->m_data.get(), [] (const T& value) -> T {
    return value;
  });
}

template<class T>
auto Storage<T>::operator=(const Storage<T> & storage) -> Storage<T>& {
  this->m_size = storage.m_size;
  this->m_data = std::make_unique<T[]>(this->m_size);
  std::transform(storage.m_data.get(), storage.m_data.get() + storage.m_size, this->m_data.get(), [] (const T &value) -> T {
    return value;
  });
  return *this;
}

template<class T>
Storage<T>::Storage(Storage<T> &&storage) : m_size(std::move(storage.m_size)), m_data(std::move(storage.m_data)) {

}

template<class T>
auto Storage<T>::operator=(Storage<T> &&storage) -> Storage<T>& {
  if (this != &storage) {
    this->m_size = std::move(storage.m_size);
    this->m_data = std::move(storage.m_data);
  }

  return *this;
}
template <class T> 
auto Storage<T>::fill(T value) -> void {
  std::fill(this->m_data.get(), this->m_data.get() + this->m_size, value);
}


template<class T>
auto Storage<T>::element_wise(const std::function<T(T)> &trans) -> void {
  std::transform(this->m_data.get(), this->m_data.get() + this->m_size, this->m_data.get(), trans);
}

template <class T> 
auto Storage<T>::operator[](size_t index) -> T & {
  assert(index < this->m_size);
  return *(this->m_data.get() + index);
}

template<class T>
auto Storage<T>::operator+(T elem) -> void {
  std::transform(this->m_data.get(), this->m_data.get() + this->m_size, this->m_data.get(), [&elem] (T value) {
    return value + elem;
  });
}
template<class T>
auto Storage<T>::operator-(T elem) -> void {
  std::transform(this->m_data.get(), this->m_data.get() + this->m_size, this->m_data.get(), [&elem] (T value) {
    return value - elem;
  });
}


template<class T>
auto Storage<T>::operator*(T scale) -> void {
  std::transform(this->m_data.get(), this->m_data.get() + this->m_size, this->m_data.get(), [&scale] (T value) {
    return value * scale;
  });
}

template<class T>
auto Storage<T>::operator/(T scale) -> void {
  std::transform(this->m_data.get(), this->m_data.get() + this->m_size, this->m_data.get(), [&scale] (T value) {
    return value / scale;
  });
}

template class Storage<int>;
template class Storage<float>;
} // namespace cpptorch