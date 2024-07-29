#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// for testing, error margin doesn't need to be that small
static constexpr float ERROR_MARGIN = 0.00001;

template <typename T>
void print_vec(const std::vector<T> &vec) {
    std::cout << "[";
    for (const T &e : vec) {
        std::cout << e << " ";
    }
    std::cout << "\b]" << std::endl;
}

template <typename T>
void check_equal_vecs(const std::vector<T> &u, const std::vector<T> &v) {
    assert(u.size() == v.size() && "Vectors don't have the same size");

    for (int i = 0; i < u.size(); i++) {
        assert(fabs(u[i] - v[i]) < ERROR_MARGIN && "Incorrect element");
    }
}