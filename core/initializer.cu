// This file implements multiple Initializer classes, meant to initialize
// neural network parameters.
//
// Further notes on these initalizers can be found here:
// https://pytorch.org/docs/stable/nn.init.html

#include "initializer.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace nnv2 {

// Fills the underlying vector with values using specified normal distribution
static void normal_init(std::vector<float> &data, float s) {
    unsigned seed =
        (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    std::normal_distribution<float> dist(0.f, s);
    std::for_each(data.begin(), data.end(), [&](float &e) { e = dist(rng); });
}

// Fills the underlying vector with values using specified unifrom distribution
static void uniform_init(std::vector<float> &data, float r) {
    unsigned seed =
        (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<float> dist(-r, r);
    std::for_each(data.begin(), data.end(), [&](float &e) { e = dist(rng); });
}

void LecunNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(1.f / fan_in);
    normal_init(a->get_vec(), s);
}

void XavierNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(2.f / (fan_in + fan_out));
    normal_init(a->get_vec(), s);
}

void KaimingNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(2.f / fan_in);
    normal_init(a->get_vec(), s);
}

void SimplyNormal::initialize(Array *a, int fan_in, int fan_out) const {
    normal_init(a->get_vec(), 0.1f);
}

void LecunUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(1.f / fan_in);
    uniform_init(a->get_vec(), r);
}

void XavierUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(6.f / (fan_in + fan_out));
    uniform_init(a->get_vec(), r);
}

void KaimingUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(6.f / fan_in);
    uniform_init(a->get_vec(), r);
}

void SimplyUniform::initialize(Array *a, int fan_in, int fan_out) const {
    uniform_init(a->get_vec(), 0.01f);
}

} // namespace nnv2