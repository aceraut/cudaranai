#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

namespace nnv2 {

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_DIM = 16;
constexpr float EPS = 1e-8;

// Check macros
#define CHECK_EQ(val1, val2, message)                                          \
    do {                                                                       \
        if ((val1) != (val2)) {                                                \
            std::cerr << __FILE__ << "(" << __LINE__ << "): " << (message)     \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_COND(condition, message)                                         \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << __FILE__ << "(" << __LINE__ << "): " << (message)     \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(condition)                                                  \
    do {                                                                       \
        cudaError_t error = condition;                                         \
        CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error));               \
    } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define RAW_PTR(vec) (thrust::raw_pointer_cast(vec.data()))

// array.cc
// Array object similar to numpy array
class Array {
public:
    explicit Array(const std::vector<int> &_shape);
    explicit Array(const std::vector<int> &_shape, float value);
    explicit Array(const std::vector<int> &_shape,
                   const std::vector<float> &_vec);

    Array(const Array &other);
    Array(Array &&other);
    Array &operator=(const Array &other);
    Array &operator=(Array &&other);

    void reshape(const std::vector<int> &_shape);
    void resize(const std::vector<int> &_shape);

    void zero();

    thrust::device_vector<float> &get_vec() { return vec; }
    const thrust::device_vector<float> &get_vec() const { return vec; }

    const std::vector<int> &get_shape() const { return shape; }

private:
    void check_shape();

    thrust::device_vector<float> vec;
    std::vector<int> shape;
};

// Initializes Array object inside smart pointer
void set_array_ptr(std::unique_ptr<Array> &ptr, const std::vector<int> &shape);

// As several functions in the training process require temporary Array objects,
// and they can be called multiple times if the size of train data is large,
// it's better to cache these temporary Array objects instead of creating new
// ones on every call.
using ArrayMap = std::unordered_map<std::string, std::unique_ptr<Array>>;

void set_array_cache(ArrayMap &map, std::string key,
                     const std::vector<int> &shape);

// A pair of weight array and gradient array, used by Optimizer to recalculate
// the weight.
using Param = std::pair<Array *, Array *>;

// Math operations on Array objects. Defined in mathfunc.cc

// Element-wise addition of 2 arrays
void func_add(Array *output, const Array *input1, const Array *input2);
// Element-wise addition of an array and a scalar
void func_add(Array *output, const Array *input, float value);

// Element-wise subtraction of 2 arrays
void func_sub(Array *output, const Array *input1, const Array *input2);
// Element-wise subtraction of an array and a scalar
void func_sub(Array *output, const Array *input, float value);

// Element-wise multiplication of 2 arrays
// See func_matmul() for the dot product of 2 arrays
void func_mul(Array *output, const Array *input1, const Array *input2);
// Element-wise multiplication of an array and a scalar
void func_mul(Array *output, const Array *input, float value);

// Element-wise division of 2 arrays
void func_div(Array *output, const Array *input1, const Array *input2);

// Element-wise natual logarithm of 2 arrays
void func_log(Array *output, const Array *input);

// Matrix multiplication or dot product of two arrays
// `broadcast` tells the function which input array needs replicating to match
// with the other input's shape constraints. If broadcast is 0, no replication
// is needed.
void func_matmul(Array *output, const Array *input1, const Array *input2,
                 int broadcast = 0);

// Matrix transpose
void func_transpose(Array *output, const Array *input);

// Sum of array elements over an axis
// `reduce` tells the function that the output must have that axis removed
void func_sum(Array *output, const Array *input, int axis, bool reduce = true);

// Mean value of array elements over an axis
// `reduce` tells the function that the output must have that axis removed
void func_mean(Array *output, const Array *input, int axis, bool reduce = true);

} // namespace nnv2