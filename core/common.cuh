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
constexpr float EPS = 1e-8;

// Used in matmul_lvl1 and transpose kernels
constexpr int TILE_DIM = 32;

// Used in matmul kernel
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

// Type definitions
using VecType = thrust::device_vector<float>;
using ShapeType = std::vector<int>;

class Array;

// Temporary Array map to cache local Array objects
using ArrayMap = std::unordered_map<std::string, std::unique_ptr<Array>>;

// A pair of weight array and gradient array, used by Optimizer to recalculate
// the weight.
using Param = std::pair<Array *, Array *>;

// Array object similar to numpy array, implemented in array.cu
class Array {
public:
    explicit Array(const ShapeType &_shape);
    explicit Array(const ShapeType &_shape, float value);
    explicit Array(const ShapeType &_shape, const VecType &_vec);

    Array(const Array &other);
    Array(Array &&other);
    Array &operator=(const Array &other);
    Array &operator=(Array &&other);

    void reshape(const ShapeType &_shape);
    void resize(const ShapeType &_shape);

    void zero();

    VecType &get_vec() { return vec; }
    const VecType &get_vec() const { return vec; }

    const ShapeType &get_shape() const { return shape; }

private:
    void check_shape();

    VecType vec;
    ShapeType shape;
};

// Helper functions, implemented in utils.cu.
namespace utils {

// Initializes Array object inside smart pointer
void set_array_ptr(std::unique_ptr<Array> &ptr, const std::vector<int> &shape);

// Since several functions in the training process require temporary Array
// objects, and these functions may be called multiple times when the training
// data is large, it's more efficient to cache these temporary Array objects
// instead of creating new ones for each call.
void set_array_cache(ArrayMap &map, std::string key,
                     const std::vector<int> &shape);

// Calculates rounded up decimal quotient of two integers
int div_ceil(int a, int b);

} // namespace utils

// Math operations for Array objects, implemented in mathops.cu.
namespace ops {

// Element-wise addition of 2 arrays
void add(Array *output, const Array *input1, const Array *input2);
// Element-wise addition of an array and a scalar
void add(Array *output, const Array *input, float value);

// Element-wise subtraction of 2 arrays
void subtract(Array *output, const Array *input1, const Array *input2);
// Element-wise subtraction of an array and a scalar
void subtract(Array *output, const Array *input, float value);

// Element-wise multiplication of 2 arrays
// See func_matmul() for the dot product of 2 arrays
void multiply(Array *output, const Array *input1, const Array *input2);
// Element-wise multiplication of an array and a scalar
void multiply(Array *output, const Array *input, float value);

// Element-wise division of 2 arrays
void divide(Array *output, const Array *input1, const Array *input2);

// Element-wise natual logarithm of 2 arrays
void log(Array *output, const Array *input);

// Matrix multiplication or dot product of two arrays
// `broadcast` tells the function which input array needs replicating to match
// with the other input's shape constraints. If broadcast is 0, no replication
// is needed.
void matmul(Array *output, const Array *input1, const Array *input2,
            int broadcast = 0);

// Matrix transpose
void transpose(Array *output, const Array *input);

// Sum of array elements over an axis
// `reduce` tells the function that the output must have that axis removed.
void sum(Array *output, const Array *input, int axis, bool reduce = true);

// Mean value of array elements over an axis
// `reduce` tells the function that the output must have that axis removed.
void mean(Array *output, const Array *input, int axis, bool reduce = true);

} // namespace ops

// Macros

// Assertion macros
#define CHECK_EQ(val1, val2, message)                                          \
    do {                                                                       \
        if ((val1) != (val2)) {                                                \
            std::cerr << __FILE__ << "(" << __LINE__ << "): " << (message)     \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_COND(statement, message)                                         \
    do {                                                                       \
        if (!(statement)) {                                                    \
            std::cerr << __FILE__ << "(" << __LINE__ << "): " << (message)     \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Macro used to check for errors after a CUDA kernel call
#define CUDA_CHECK(statement)                                                  \
    do {                                                                       \
        cudaError_t error = statement;                                         \
        CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error));               \
    } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// Macro used to get raw pointer from Thrust device_vector
#define RAW_PTR(vec) (thrust::raw_pointer_cast(vec.data()))

// Instead of completely unrolling the loop when parallelizing the computation
// by assigning each independent iteration to a thread, we use a grid-stride
// loop in the kernel. This approach allows the kernel to loop over the data
// array one grid-size at a time, making it easier to scale the number of
// threads while maintaining parallelism.
//
// Reference:
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#define CUDA_GRID_STRIDE_LOOP(i, n)                                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                 \
         i += gridDim.x * blockDim.x)

} // namespace nnv2