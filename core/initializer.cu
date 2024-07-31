// This file implements multiple Initializer classes, meant to initialize
// neural network parameters.
//
// Further notes on these initalizers can be found here:
// https://pytorch.org/docs/stable/nn.init.html

#include "common.cuh"
#include "initializer.cuh"

#include <chrono>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

namespace nnv2 {

// kernel to initialize data with normal distribution
__global__ void normal_init_kernel(int size, float *vec, float mean,
                                   float stddev, unsigned seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        vec[idx] = mean + stddev * curand_normal(&state);
    }
}

static void normal_init(thrust::device_vector<float> &vec, float s) {
    float *vec_raw = RAW_PTR(vec);
    unsigned seed =
        (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();

    int size = vec.size();
    int grid_size = ceil(float(size) / BLOCK_SIZE);

    // Launch kernel
    normal_init_kernel<<<grid_size, BLOCK_SIZE>>>(size, vec_raw, 0, s, seed);
    CUDA_POST_KERNEL_CHECK;
}

// kernel to initialize data with uniform distribution
__global__ void uniform_init_kernel(int size, float *vec, float a, float b,
                                    unsigned seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        vec[idx] = a + (b - a) * curand_uniform(&state);
    }
}

static void uniform_init(thrust::device_vector<float> &vec, float r) {
    float *vec_raw = RAW_PTR(vec);
    unsigned seed =
        (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();

    int size = vec.size();
    int grid_size = ceil(float(size) / BLOCK_SIZE);

    // Launch kernel
    uniform_init_kernel<<<grid_size, BLOCK_SIZE>>>(size, vec_raw, -r, r, seed);
    CUDA_POST_KERNEL_CHECK;
}

void LecunNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(1.0 / fan_in);
    normal_init(a->get_vec(), s);
}

void XavierNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(2.0 / (fan_in + fan_out));
    normal_init(a->get_vec(), s);
}

void KaimingNormal::initialize(Array *a, int fan_in, int fan_out) const {
    float s = sqrtf(2.0 / fan_in);
    normal_init(a->get_vec(), s);
}

void SimplyNormal::initialize(Array *a, int fan_in, int fan_out) const {
    normal_init(a->get_vec(), 0.1);
}

void LecunUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(1.0 / fan_in);
    uniform_init(a->get_vec(), r);
}

void XavierUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(6.0 / (fan_in + fan_out));
    uniform_init(a->get_vec(), r);
}

void KaimingUniform::initialize(Array *a, int fan_in, int fan_out) const {
    float r = sqrtf(6.0 / fan_in);
    uniform_init(a->get_vec(), r);
}

void SimplyUniform::initialize(Array *a, int fan_in, int fan_out) const {
    uniform_init(a->get_vec(), 0.01);
}

} // namespace nnv2