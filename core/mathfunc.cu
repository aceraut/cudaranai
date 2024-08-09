// This file implements math operations on Array objects.

#include "common.cuh"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace nnv2 {

void func_add(Array *output, const Array *input1, const Array *input2) {
    int input1_size = input1->get_vec().size();
    int input2_size = input2->get_vec().size();
    int output_size = output->get_vec().size();

    CHECK_EQ(input1_size, input2_size,
             "func_add: size mismatch between inputs");
    CHECK_EQ(output_size, input1_size,
             "func_add: size mismatched between input and output");

    thrust::transform(input1->get_vec().begin(), input1->get_vec().end(),
                      input2->get_vec().begin(), output->get_vec().begin(),
                      thrust::plus<float>());
}

void func_add(Array *output, const Array *input, float value) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "func_add: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [value] __device__(float x) { return x + value; });
}

void func_sub(Array *output, const Array *input1, const Array *input2) {
    int input1_size = input1->get_vec().size();
    int input2_size = input2->get_vec().size();
    int output_size = output->get_vec().size();

    CHECK_EQ(input1_size, input2_size,
             "func_sub: size mismatch between inputs");
    CHECK_EQ(output_size, input1_size,
             "func_sub: size mismatch between input and outputs");

    thrust::transform(input1->get_vec().begin(), input1->get_vec().end(),
                      input2->get_vec().begin(), output->get_vec().begin(),
                      thrust::minus<float>());
}

void func_sub(Array *output, const Array *input, float value) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "func_sub: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [value] __device__(float x) { return x - value; });
}

void func_mul(Array *output, const Array *input1, const Array *input2) {
    int input1_size = input1->get_vec().size();
    int input2_size = input2->get_vec().size();
    int output_size = output->get_vec().size();

    CHECK_EQ(input1_size, input2_size,
             "func_mul: size mismatch between inputs");
    CHECK_EQ(output_size, input1_size,
             "func_mul: size mismatch between input and outputs");

    thrust::transform(input1->get_vec().begin(), input1->get_vec().end(),
                      input2->get_vec().begin(), output->get_vec().begin(),
                      thrust::multiplies<float>());
}

void func_mul(Array *output, const Array *input, float value) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "func_mul: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [value] __device__(float x) { return x * value; });
}

void func_div(Array *output, const Array *input1, const Array *input2) {
    int input1_size = input1->get_vec().size();
    int input2_size = input2->get_vec().size();
    int output_size = output->get_vec().size();

    CHECK_EQ(input1_size, input2_size,
             "func_div: size mismatch between inputs");
    CHECK_EQ(output_size, input1_size,
             "func_div: size mismatch between input and outputs");

    thrust::transform(input1->get_vec().begin(), input1->get_vec().end(),
                      input2->get_vec().begin(), output->get_vec().begin(),
                      thrust::divides<float>());
}

void func_log(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "func_log: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [] __device__(float e) { return logf(e); });
}

__global__ void func_matmul_kernel(float *output, const float *input1,
                                   const float *input2, int m, int n, int k,
                                   int broadcast) {
    __shared__ float input1_tile[TILE_DIM][TILE_DIM];
    __shared__ float input2_tile[TILE_DIM][TILE_DIM];

    // calculate offsets of the matrices
    int batch_idx = blockIdx.z;
    if (broadcast != 1) {
        input1 += batch_idx * m * k;
    }
    if (broadcast != 2) {
        input2 += batch_idx * k * n;
    }
    output += batch_idx * m * n;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_DIM + tx;
    int col = by * TILE_DIM + ty;

    // loop over input tiles to calculate the dot value
    float value = 0;

    for (int i = 0; i < (int)ceil((float)k / TILE_DIM); i++) {
        // load input tiles to shared memory
        if (row < m && i * TILE_DIM + ty < k) {
            input1_tile[tx][ty] = input1[row * k + i * TILE_DIM + ty];
        } else {
            input1_tile[tx][ty] = 0;
        }
        if (col < n && i * TILE_DIM + tx < k) {
            input2_tile[tx][ty] = input2[(i * TILE_DIM + tx) * n + col];
        } else {
            input2_tile[tx][ty] = 0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_DIM; j++) {
            value += input1_tile[tx][j] * input2_tile[j][ty];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        output[row * n + col] = value;
    }
}

void func_matmul(Array *output, const Array *input1, const Array *input2,
                 int broadcast) {
    // Performs matrix multiplication with two modes based on `broadcast` value:
    //
    // `broadcast == 0` (default):
    // - For 2D inputs: single matrix multiplication.
    // - For higher dimensions: batch matrix multiplication on corresponding
    // matrices.
    //
    // `broadcast == 1 or 2`:
    // - Batch matrix multiplication between a batch A and a single matrix B.
    // - If `broadcast == 1`, B is the first input; if `broadcast == 2`, B is
    // the second input.

    CHECK_COND(input1->get_shape().size() > 1,
               "func_matmul: shape error at first input");
    CHECK_COND(input2->get_shape().size() > 1,
               "func_matmul: shape error at second input");
    CHECK_COND(output->get_shape().size() > 1,
               "func_matmul: shape error at output");

    // additional dimension check for broadcast case
    if (broadcast == 1) {
        CHECK_EQ(input1->get_shape().size(), 2,
                 "func_matmul: shape error at first input");
    } else if (broadcast == 2) {
        CHECK_EQ(input2->get_shape().size(), 2,
                 "func_matmul: shape error at second input");
    }

    // calculate batch size and validate
    int batch_size = std::accumulate(output->get_shape().begin(),
                                     output->get_shape().end() - 2, 1,
                                     std::multiplies<int>());
    int bs_input1 = std::accumulate(input1->get_shape().begin(),
                                    input1->get_shape().end() - 2, 1,
                                    std::multiplies<int>());
    int bs_input2 = std::accumulate(input2->get_shape().begin(),
                                    input2->get_shape().end() - 2, 1,
                                    std::multiplies<int>());

    if (broadcast != 1) {
        CHECK_EQ(batch_size, bs_input1, "func_matmul: batch size mismatch");
    }
    if (broadcast != 2) {
        CHECK_EQ(batch_size, bs_input2, "func_matmul: batch size mismatch");
    }

    // validate matrix dimension
    int m = *(input1->get_shape().rbegin() + 1);
    int k = *(input1->get_shape().rbegin());
    int n = *(input2->get_shape().rbegin());
    int input2_h = *(input2->get_shape().rbegin() + 1);
    int output_h = *(output->get_shape().rbegin() + 1);
    int output_w = *(output->get_shape().rbegin());

    CHECK_EQ(k, input2_h, "func_matmul: shape mismatch between inputs");
    CHECK_EQ(m, output_h,
             "func_matmul: shape mismatch between first input and output");
    CHECK_EQ(n, output_w,
             "func_matmul: shape mismatch between second input and output");

    dim3 grid_dim(ceil((float)n / TILE_DIM), ceil((float)m / TILE_DIM),
                  batch_size);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input1_raw = RAW_PTR(input1->get_vec());
    const float *input2_raw = RAW_PTR(input2->get_vec());

    func_matmul_kernel<<<grid_dim, block_dim>>>(output_raw, input1_raw,
                                                input2_raw, m, n, k, broadcast);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void func_transpose_kernel(float *output, const float *input, int m,
                                      int n) {
    __shared__ float input_tile[TILE_DIM][TILE_DIM];

    int batch_idx = blockIdx.z;
    input += batch_idx * m * n;
    output += batch_idx * n * m;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_DIM + tx;
    int col = by * TILE_DIM + ty;

    if (row < m && col < n) {
        input_tile[tx][ty] = input[row * n + col];
        __syncthreads();
        output[col * m + row] = input_tile[tx][ty];
    }
}

void func_transpose(Array *output, const Array *input) {
    // Performs matrix tranpose
    //
    // If input have more than 2 dimensions, it performs batch matrix transpose
    // which requires output to have the same batch size as the input array

    // check if the dimensions are at least 2
    CHECK_COND(input->get_shape().size() > 1,
               "func_transpose: shape error at input");
    CHECK_COND(output->get_shape().size() > 1,
               "func_transpose: shape error at output");

    // calculate batch size and validate
    int batch_size = std::accumulate(output->get_shape().begin(),
                                     output->get_shape().end() - 2, 1,
                                     std::multiplies<int>());
    int bs_input = std::accumulate(input->get_shape().begin(),
                                   input->get_shape().end() - 2, 1,
                                   std::multiplies<int>());
    CHECK_EQ(batch_size, bs_input, "func_transpose: batch size mismatch");

    // validate matrix dimension
    int m = *(input->get_shape().rbegin() + 1);
    int n = *(input->get_shape().rbegin());
    int output_h = *(output->get_shape().rbegin() + 1);
    int output_w = *(output->get_shape().rbegin());

    CHECK_EQ(m, output_w,
             "func_transpose: shape mismatch between input and output");
    CHECK_EQ(n, output_h,
             "func_transpose: shape mismatch between input and output");

    // launch kernels
    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    dim3 grid_dim(ceil((float)n / TILE_DIM), ceil((float)m / TILE_DIM),
                  batch_size);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    func_transpose_kernel<<<grid_dim, block_dim>>>(output_raw, input_raw, m, n);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void func_sum_kernel(int size, float *output, const float *input,
                                int axis_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int base = (idx / stride) * axis_size * stride + (idx % stride);

        float value = 0;
        for (int i = 0; i < axis_size; i++) {
            value += input[base + i * stride];
        }
        output[idx] = value;
    }
}

void func_sum(Array *output, const Array *input, int axis, bool reduce) {
    // Calculates sum of array elements along a given axis
    //
    // The parameter `reduce` indicates whether the dimension at `axis`
    // in input array is removed in the output

    CHECK_COND(axis >= 0,
               "func_sum: support for negative axis isn't implemented");
    CHECK_COND(axis < input->get_shape().size(),
               "func_sum: axis is out of bound");

    // validate output shape
    // if `reduce` is true, remove the element at `axis` from output shape
    std::vector<int> output_shape = input->get_shape();
    if (reduce && output_shape.size() > 1) {
        output_shape.erase(output_shape.begin() + axis);
    } else {
        output_shape[axis] = 1;
    }
    CHECK_EQ(output->get_shape(), output_shape,
             "func_sum: shape error at output");

    // launch kernels
    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    int output_size = output->get_vec().size();
    int axis_size = input->get_shape()[axis];
    int stride =
        std::accumulate(input->get_shape().begin() + axis + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    int grid_size = ceil((float)output_size / BLOCK_SIZE);
    func_sum_kernel<<<grid_size, BLOCK_SIZE>>>(output_size, output_raw,
                                               input_raw, axis_size, stride);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void func_mean_kernel(int size, float *output, const float *input,
                                 int axis_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int base = (idx / stride) * axis_size * stride + (idx % stride);

        float value = 0;
        for (int i = 0; i < axis_size; i++) {
            value += input[base + i * stride];
        }
        output[idx] = value / axis_size;
    }
}

void func_mean(Array *output, const Array *input, int axis, bool reduce) {
    // Calculates mean value of array elements along a given axis
    //
    // The parameter `reduce` indicates whether the dimension at `axis`
    // in input array is removed in the output

    CHECK_COND(axis >= 0,
               "func_mean: support for negative axis isn't implemented");
    CHECK_COND(axis < input->get_shape().size(),
               "func_mean: axis is out of bound");

    // validate output shape
    // if `reduce` is true, remove the element at `axis` from output shape
    std::vector<int> output_shape = input->get_shape();
    if (reduce && output_shape.size() > 1) {
        output_shape.erase(output_shape.begin() + axis);
    } else {
        output_shape[axis] = 1;
    }
    CHECK_EQ(output->get_shape(), output_shape,
             "func_mean: shape error at output");

    // launch kernels
    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    int output_size = output->get_vec().size();
    int axis_size = input->get_shape()[axis];
    int stride =
        std::accumulate(input->get_shape().begin() + axis + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    int grid_size = ceil((float)output_size / BLOCK_SIZE);
    func_mean_kernel<<<grid_size, BLOCK_SIZE>>>(output_size, output_raw,
                                                input_raw, axis_size, stride);
    CUDA_POST_KERNEL_CHECK;
}

} // namespace nnv2