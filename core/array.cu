#include "common.cuh"

#include <utility>

#include <thrust/fill.h>

namespace nnv2 {

Array::Array(const ShapeType &_shape, float value) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  vec.resize(size, value);
}

Array::Array(const ShapeType &_shape, const VecType<float> &_vec)
    : shape(_shape), vec(_vec.begin(), _vec.end()) {
  check_shape();
}

Array::Array(const Array &other) { *this = other; }

Array::Array(Array &&other) { *this = std::move(other); }

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

void Array::zero() { thrust::fill(vec.begin(), vec.end(), 0.0); }

void Array::reshape(const ShapeType &_shape) {
  shape = _shape;
  check_shape();
}

void Array::resize(const ShapeType &_shape) {
  shape = _shape;
  
  int size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  if (size != vec.size()) {
    vec.resize(size);
  }
}

void Array::check_shape() {
  int size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  CHECK_EQ(size, vec.size(), "Array: mismatch between array shape and size");
}

} // namespace nnv2