#include "common.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace nnv2 {

Array::Array(const std::vector<int> &_shape) : shape(_shape) {
    int size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    vec.resize(size);
}

Array::Array(const std::vector<int> &_shape, float _value) : shape(_shape) {
    int size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    vec.resize(size, _value);
}

Array::Array(const std::vector<int> &_shape, const std::vector<float> &_vec)
    : shape(_shape), vec(_vec.begin(), _vec.end()) {
    check_shape();
}

Array::Array(const Array &other) {
    *this = other;
}

Array::Array(Array &&other) {
    *this = std::move(other);
}

Array &Array::operator=(const Array &other) {
    if (this != &other) {
        shape = other.shape;
        vec = other.vec;
    }
    return *this;
}

Array &Array::operator=(Array &&other) {
    if (this != &other) {
        shape = std::move(other.shape);
        vec = std::move(other.vec);
    }
    return *this;
}

void Array::zero() {
    std::fill(vec.begin(), vec.end(), 0.0);
}

void Array::reshape(const std::vector<int> &_shape) {
    shape = _shape;
    check_shape();
}

void Array::resize(const std::vector<int> &_shape) {
    shape = _shape;
    int size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (size != vec.size()) {
        vec.resize(size);
    }
}

void Array::check_shape() {
    int size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    CHECK_EQ(size, vec.size(), "Array: shape mismatched with size");
}

//
// functions that help initialize Array objects from smart pointers or ArrayMap
void set_array_ptr(std::unique_ptr<Array> &ptr, const std::vector<int> &shape) {
    if (ptr.get() == nullptr) {
        ptr.reset(new Array(shape));
    } else {
        if (ptr->get_shape() != shape) {
            ptr->resize(shape);
        }
        ptr->zero();
    }
}

void set_array_cache(ArrayMap &map, std::string key,
                     const std::vector<int> &shape) {
    if (map.find(key) == map.end()) {
        map[key] = std::make_unique<Array>(shape);
    }
    set_array_ptr(map[key], shape);
}

} // namespace nnv2