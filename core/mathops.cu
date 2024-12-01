#include "common.cuh"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace nnv2 {

__global__ void matmul_kernel(float *output, const float *input1,
                              const float *input2, int m, int n, int k,
                              int broadcast) {
    __shared__ float input1_tile[TILE_DIM][TILE_DIM];
    __shared__ float input2_tile[TILE_DIM][TILE_DIM];

    // Calculate offsets of the matrices
    int batch_idx = blockIdx.z;
    if (broadcast != 1) {
        input1 += batch_idx * m * k;
    }
    if (broadcast != 2) {
        input2 += batch_idx * k * n;
    }
    output += batch_idx * m * n;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Loop over input tiles to calculate the dot value
    float value = 0;
    int tile_count = (k + TILE_DIM - 1) / TILE_DIM;

    for (int i = 0; i < tile_count; i++) {
        // Load input tiles to shared memory
        if (row < m && i * TILE_DIM + tx < k) {
            input1_tile[ty][tx] = input1[row * k + i * TILE_DIM + tx];
        } else {
            input1_tile[ty][tx] = 0;
        }
        if (col < n && i * TILE_DIM + ty < k) {
            input2_tile[ty][tx] = input2[(i * TILE_DIM + ty) * n + col];
        } else {
            input2_tile[ty][tx] = 0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_DIM; j++) {
            value += input1_tile[ty][j] * input2_tile[j][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        output[row * n + col] = value;
    }
}

// Another implementation (which is really slow and not as precise)
__global__ void matmul_kernel_exp(float *output, const float *input1,
                                  const float *input2, int m, int n, int k,
                                  int broadcast) {
    __shared__ float block1[BM][BK];
    __shared__ float block2[BK][BN];

    float thread_output[TM * TN] = {0.0};
    float reg1[TM] = {0.0};
    float reg2[TM] = {0.0};

    // Calculate offsets of the matrices
    const int batch_idx = blockIdx.z;
    if (broadcast != 1) {
        input1 += batch_idx * m * k;
    }
    if (broadcast != 2) {
        input2 += batch_idx * k * n;
    }
    output += batch_idx * m * n;

    // Indices of current block in output matrix
    const int bx = blockIdx.y;
    const int by = blockIdx.x;

    // Indices of current grid in said output matrix block
    const int tx = threadIdx.x / (BN / TN);
    const int ty = threadIdx.x % (BN / TN);

    // Number of threads per block
    const int nthreads = blockDim.x;

    // In phase 1, we populate block1 by having all threads traverse (BM, BK)
    // grids in the first input through a (nthreads/BK, BK) window, and populate
    // block2 by having all threads traverse (BK, BN) grids in the second input
    // through a (nthreads/BN, BN) window.
    // The two constants are row strides on each traversal step.
    const int win1_stride = nthreads / BK;
    const int win2_stride = nthreads / BN;

    // Coordinates of the 2 current entries in those two windows
    const int x_win1 = threadIdx.x / BK;
    const int y_win1 = threadIdx.x % BK;
    const int x_win2 = threadIdx.x / BN;
    const int y_win2 = threadIdx.x % BN;

    for (int block_offset = 0; block_offset < k; block_offset += BK) {
        // Populate block1
        for (int win_offset = 0; win_offset < BM; win_offset += win1_stride) {
            const int x_block1 = x_win1 + win_offset;
            const int y_block1 = y_win1;
            const int x_input1 = x_block1 + bx * BM;
            const int y_input1 = y_block1 + block_offset;

            block1[x_block1][y_block1] = (x_input1 < m && y_input1 < k)
                                             ? input1[x_input1 * k + y_input1]
                                             : 0.0;
        }

        // Populate block2
        for (int win_offset = 0; win_offset < BK; win_offset += win2_stride) {
            const int x_block2 = x_win2 + win_offset;
            const int y_block2 = y_win2;
            const int x_input2 = x_block2 + block_offset;
            const int y_input2 = y_block2 + by * BN;

            block1[x_block2][y_block2] = (x_input2 < k && y_input2 < n)
                                             ? input1[x_input2 * n + y_input2]
                                             : 0.0;
        }
        __syncthreads();

        // In phase 2, each thread calculates a (TM, TN) tile in output matrix,
        // corresponding to (TM, K) grid in the first input and (K, TN) grid in
        // the second input. The outer loop traverses through K/BK blocks in
        // both inputs so we only need to load a (TM, BK) grid in the first
        // input and a (BK, TN) grid in the second one.
        // The output value can be accumulated through BK (and K in broad)
        // so we can simply load TM values on the first grid and TN values
        // on the second grid.
        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < TM; j++) {
                reg1[j] = block1[tx * TM + j][i];
            }
            for (int j = 0; j < TN; j++) {
                reg2[j] = block2[i][ty * TN + j];
            }
            for (int x = 0; x < TM; x++) {
                for (int y = 0; y < TN; y++) {
                    thread_output[x * TN + y] += reg1[x] * reg2[y];
                }
            }
        }
        __syncthreads();
    }

    for (int x_tile = 0; x_tile < TM; x_tile++) {
        for (int y_tile = 0; y_tile < TN; y_tile++) {
            const int x = bx * BM + tx * TM + x_tile;
            const int y = by * BN + ty * TN + y_tile;
            if (x < m && y < n) {
                output[x * n + y] += thread_output[x_tile * TN + y_tile];
            }
        }
    }
}

__global__ void transpose_kernel(float *output, const float *input, int m,
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

__global__ void sum_kernel(int size, float *output, const float *input,
                           int axis_size, int stride) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        int base = (idx / stride) * axis_size * stride + (idx % stride);

        float value = 0;
        for (int i = 0; i < axis_size; i++) {
            value += input[base + i * stride];
        }
        output[idx] = value;
    }
}

__global__ void mean_kernel(int size, float *output, const float *input,
                            int axis_size, int stride) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        int base = (idx / stride) * axis_size * stride + (idx % stride);

        float value = 0;
        for (int i = 0; i < axis_size; i++) {
            value += input[base + i * stride];
        }
        output[idx] = value / axis_size;
    }
}

namespace ops {

void add(Array *output, const Array *input1, const Array *input2) {
    const VecType &input1_vec = input1->get_vec();
    const VecType &input2_vec = input2->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(input1_vec.size(), input2_vec.size(),
             "ops::add: size mismatch between inputs");
    CHECK_EQ(output_vec.size(), input1_vec.size(),
             "ops::add: size mismatched between input and output");

    thrust::transform(input1_vec.begin(), input1_vec.end(), input2_vec.begin(),
                      output_vec.begin(), thrust::plus<float>());
}

void add(Array *output, const Array *input, float value) {
    const VecType &input_vec = input->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(output_vec.size(), input_vec.size(),
             "ops::add: size mismatch between input and output");

    thrust::transform(input_vec.begin(), input_vec.end(), output_vec.begin(),
                      [value] __device__(float x) { return x + value; });
}

void subtract(Array *output, const Array *input1, const Array *input2) {
    const VecType &input1_vec = input1->get_vec();
    const VecType &input2_vec = input2->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(input1_vec.size(), input2_vec.size(),
             "ops::subtract: size mismatch between inputs");
    CHECK_EQ(output_vec.size(), input1_vec.size(),
             "ops::subtract: size mismatch between input and outputs");

    thrust::transform(input1_vec.begin(), input1_vec.end(), input2_vec.begin(),
                      output_vec.begin(), thrust::minus<float>());
}

void subtract(Array *output, const Array *input, float value) {
    const VecType &input_vec = input->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(output_vec.size(), input_vec.size(),
             "ops::subtract: size mismatch between input and output");

    thrust::transform(input_vec.begin(), input_vec.end(), output_vec.begin(),
                      [value] __device__(float x) { return x - value; });
}

void multiply(Array *output, const Array *input1, const Array *input2) {
    const VecType &input1_vec = input1->get_vec();
    const VecType &input2_vec = input2->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(input1_vec.size(), input2_vec.size(),
             "ops::multiply: size mismatch between inputs");
    CHECK_EQ(output_vec.size(), input1_vec.size(),
             "ops::multiply: size mismatch between input and outputs");

    thrust::transform(input1_vec.begin(), input1_vec.end(), input2_vec.begin(),
                      output_vec.begin(), thrust::multiplies<float>());
}

void multiply(Array *output, const Array *input, float value) {
    const VecType &input_vec = input->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(output_vec.size(), input_vec.size(),
             "ops::multiply: size mismatch between input and output");

    thrust::transform(input_vec.begin(), input_vec.end(), output_vec.begin(),
                      [value] __device__(float x) { return x * value; });
}

void divide(Array *output, const Array *input1, const Array *input2) {
    const VecType &input1_vec = input1->get_vec();
    const VecType &input2_vec = input2->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(input1_vec.size(), input2_vec.size(),
             "ops::divide: size mismatch between inputs");
    CHECK_EQ(output_vec.size(), input1_vec.size(),
             "ops::divide: size mismatch between input and outputs");

    thrust::transform(input1_vec.begin(), input1_vec.end(), input2_vec.begin(),
                      output_vec.begin(), thrust::divides<float>());
}

void log(Array *output, const Array *input) {
    const VecType &input_vec = input->get_vec();
    VecType &output_vec = output->get_vec();

    CHECK_EQ(output_vec.size(), input_vec.size(),
             "ops::log: size mismatch between input and output");

    thrust::transform(input_vec.begin(), input_vec.end(), output_vec.begin(),
                      [] __device__(float e) { return logf(e); });
}

// Performs matrix multiplication with two modes based on `broadcast` value:
//
// `broadcast == 0` (default):
// - For 2D inputs: single matrix multiplication.
// - For higher dimensions: batch matrix multiplication on corresponding
// matrices.
//
// `broadcast == 1 or 2`:
// - Batch matrix multiplication between a batch A and a single matrix B.
// - If `broadcast == 1`, B is the first input; if `broadcast == 2`, B is the
// second input.
void matmul(Array *output, const Array *input1, const Array *input2,
            int broadcast) {
    const ShapeType &output_shape = output->get_shape();
    const ShapeType &input1_shape = input1->get_shape();
    const ShapeType &input2_shape = input2->get_shape();

    CHECK_COND(input1_shape.size() > 1,
               "ops::matmul: shape error at first input");
    CHECK_COND(input2_shape.size() > 1,
               "ops::matmul: shape error at second input");
    CHECK_COND(output_shape.size() > 1, "ops::matmul: shape error at output");

    // Additional dimension check for broadcast case
    if (broadcast == 1) {
        CHECK_EQ(input1_shape.size(), 2,
                 "ops::matmul: shape error at first input");
    } else if (broadcast == 2) {
        CHECK_EQ(input2_shape.size(), 2,
                 "ops::matmul: shape error at second input");
    }

    // Calculate batch size and validate
    int batch_size =
        std::accumulate(output_shape.begin(), output_shape.end() - 2, 1,
                        std::multiplies<int>());
    int bs_input1 =
        std::accumulate(input1_shape.begin(), input1_shape.end() - 2, 1,
                        std::multiplies<int>());
    int bs_input2 =
        std::accumulate(input2_shape.begin(), input2_shape.end() - 2, 1,
                        std::multiplies<int>());

    if (broadcast != 1) {
        CHECK_EQ(batch_size, bs_input1, "ops::matmul: batch size mismatch");
    }
    if (broadcast != 2) {
        CHECK_EQ(batch_size, bs_input2, "ops::matmul: batch size mismatch");
    }

    // Validate matrix dimension
    int m = *(input1_shape.rbegin() + 1);
    int k = *(input1_shape.rbegin());
    int n = *(input2_shape.rbegin());
    int input2_h = *(input2_shape.rbegin() + 1);
    int output_h = *(output_shape.rbegin() + 1);
    int output_w = *(output_shape.rbegin());

    CHECK_EQ(k, input2_h, "ops::matmul: shape mismatch between inputs");
    CHECK_EQ(m, output_h,
             "ops::matmul: shape mismatch between first input and output");
    CHECK_EQ(n, output_w,
             "ops::matmul: shape mismatch between second input and output");

    // Launch kernels
    dim3 grid_dim(utils::div_ceil(n, TILE_DIM), utils::div_ceil(m, TILE_DIM),
                  batch_size);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input1_raw = RAW_PTR(input1->get_vec());
    const float *input2_raw = RAW_PTR(input2->get_vec());

    matmul_kernel<<<grid_dim, block_dim>>>(output_raw, input1_raw, input2_raw,
                                           m, n, k, broadcast);
    CUDA_POST_KERNEL_CHECK;

    // dim3 grid_dim(utils::div_ceil(n, BN), utils::div_ceil(m, BM),
    //               batch_size);
    // dim3 block_dim((BM * BN) / (TM * TN));

    // float *output_raw = RAW_PTR(output->get_vec());
    // const float *input1_raw = RAW_PTR(input1->get_vec());
    // const float *input2_raw = RAW_PTR(input2->get_vec());

    // matmul_kernel_exp<<<grid_dim, block_dim>>>(
    //     output_raw, input1_raw, input2_raw, m, n, k, broadcast);
    // CUDA_POST_KERNEL_CHECK;
}

// Performs matrix tranpose. If the input has more than 2 dimensions, batch
// matrix transpose is performed, which requires output to have the same batch
// size as the input array
void transpose(Array *output, const Array *input) {
    const ShapeType &output_shape = output->get_shape();
    const ShapeType &input_shape = input->get_shape();

    // Check if the dimensions are at least 2
    CHECK_COND(input_shape.size() > 1, "ops::transpose: shape error at input");
    CHECK_COND(output_shape.size() > 1,
               "ops::transpose: shape error at output");

    // Calculate batch size and validate
    int batch_size =
        std::accumulate(output_shape.begin(), output_shape.end() - 2, 1,
                        std::multiplies<int>());
    int bs_input = std::accumulate(input_shape.begin(), input_shape.end() - 2,
                                   1, std::multiplies<int>());
    CHECK_EQ(batch_size, bs_input, "ops::transpose: batch size mismatch");

    // Validate matrix dimension
    int m = *(input_shape.rbegin() + 1);
    int n = *(input_shape.rbegin());
    int output_h = *(output_shape.rbegin() + 1);
    int output_w = *(output_shape.rbegin());

    CHECK_EQ(m, output_w,
             "ops::transpose: shape mismatch between input and output");
    CHECK_EQ(n, output_h,
             "ops::transpose: shape mismatch between input and output");

    // Launch kernels
    dim3 grid_dim(utils::div_ceil(n, TILE_DIM), utils::div_ceil(m, TILE_DIM),
                  batch_size);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    transpose_kernel<<<grid_dim, block_dim>>>(output_raw, input_raw, m, n);
    CUDA_POST_KERNEL_CHECK;
}

// Calculates sum of array elements along a given axis. The parameter `reduce`
// indicates whether the dimension at `axis` in input array is removed in the
// output.
void sum(Array *output, const Array *input, int axis, bool reduce) {
    const ShapeType &input_shape = input->get_shape();
    const ShapeType &output_shape = output->get_shape();

    CHECK_COND(axis >= 0,
               "ops::sum: support for negative axis isn't implemented");
    CHECK_COND(axis < input_shape.size(), "ops::sum: axis is out of bound");

    // Validate output shape
    // If `reduce` is true, remove the element at `axis` from output shape
    ShapeType reduced_shape = input->get_shape();
    if (reduce && input_shape.size() > 1) {
        reduced_shape.erase(reduced_shape.begin() + axis);
    } else {
        reduced_shape[axis] = 1;
    }
    CHECK_EQ(reduced_shape, output_shape, "ops::sum: shape error at output");

    // Launch kernels
    int output_size = output->get_vec().size();
    int axis_size = input_shape[axis];
    int stride = std::accumulate(input_shape.begin() + axis + 1,
                                 input_shape.end(), 1, std::multiplies<int>());
    int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    sum_kernel<<<grid_size, BLOCK_SIZE>>>(output_size, output_raw, input_raw,
                                          axis_size, stride);
    CUDA_POST_KERNEL_CHECK;
}

// Calculates mean value of array elements along a given axis. The parameter
// `reduce` indicates whether the dimension at `axis` in input array is removed
// in the output.
void mean(Array *output, const Array *input, int axis, bool reduce) {
    const ShapeType &input_shape = input->get_shape();
    const ShapeType &output_shape = output->get_shape();

    CHECK_COND(axis >= 0,
               "ops::mean: support for negative axis isn't implemented");
    CHECK_COND(axis < input_shape.size(), "ops::mean: axis is out of bound");

    // Validate output shape
    // If `reduce` is true, remove the element at `axis` from output shape
    ShapeType reduced_shape = input->get_shape();
    if (reduce && input_shape.size() > 1) {
        reduced_shape.erase(reduced_shape.begin() + axis);
    } else {
        reduced_shape[axis] = 1;
    }
    CHECK_EQ(reduced_shape, output_shape, "ops::mean: shape error at output");

    // Launch kernels
    int output_size = output->get_vec().size();
    int axis_size = input_shape[axis];
    int stride = std::accumulate(input_shape.begin() + axis + 1,
                                 input_shape.end(), 1, std::multiplies<int>());
    int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    mean_kernel<<<grid_size, BLOCK_SIZE>>>(output_size, output_raw, input_raw,
                                           axis_size, stride);
    CUDA_POST_KERNEL_CHECK;
}

} // namespace ops
} // namespace nnv2