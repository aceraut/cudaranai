#pragma once

#include <cassert>
#include <iostream>
#include <math>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

// For testing, error margin doesn't need to be that small
static constexpr float ERROR_MARGIN = 0.00001;

template <typename T>
void print_vec(const thrust::device_vector<T> &vec) {
    std::cout << "[";
    thrust::copy(vec.begin(), vec.end(),
                 std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\b]" << std::endl;
}

template <typename T>
void print_vec(const std::vector<T> &vec) {
    std::cout << "[";
    for (const T &e : vec) {
        std::cout << e << " ";
    }
    std::cout << "\b]" << std::endl;
}

template <typename T>
void check_equal_vecs(const thrust::device_vector<T> &u,
                      const std::vector<T> &v) {
    assert(u.size() == v.size() && "size mismatch between vectors");
    for (int i = 0; i < u.size(); i++) {
        assert(std::abs(u[i] - v[i]) < ERROR_MARGIN && "Incorrect element");
    }
}