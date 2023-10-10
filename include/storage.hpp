#ifndef CPPTORCH_INCLUDE_STORAGE
#define CPPTORCH_INCLUDE_STORAGE

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>

namespace cpptorch {

template<class T>
class Storage {
public:
    Storage();
    explicit Storage(size_t size);
    explicit Storage(T data[], size_t size);
    Storage(const Storage<T>&);
    auto operator=(const Storage<T>&) -> Storage<T>&;

    Storage(Storage<T>&&);
    auto operator=(Storage<T>&&) -> Storage<T>&;
public:
    auto inline size() -> size_t {
        return m_size;
    }
public:
    auto fill(T value) -> void;
    auto element_wise(const std::function<T(T)>&) -> void;
public:
    auto operator[](size_t index) -> T&;
    // inplace broadcast
    auto operator+(T elem) -> void;
    auto operator-(T elem) -> void;
    auto operator*(T scale) -> void;
    auto operator/(T scale) -> void;
private:
    std::unique_ptr<T[]> m_data;
    size_t m_size;
};

}

#endif