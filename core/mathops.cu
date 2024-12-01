#include "common.cuh"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace nnv2 {

__global__ void matmul_kernel_lv1(float *output, const float *input1,
                                  const float *input2, int m, int n, int k,
                                  int broadcast) {
    __shared__ float input1_tile[TILE_DIM][TILE_DIM];
    __shared__ float input2_tile[TILE_DIM][TILE_DIM];

    // Calculate offsets of the matrices
    const int batch_idx = blockIdx.z;
    if (broadcast != 1) {
        input1 += batch_idx * m * k;
    }
    if (broadcast != 2) {
        input2 += batch_idx * k * n;
    }
    output += batch_idx * m * n;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;

    // Loop over input tiles to calculate the dot value
    float value = 0;
    const int tile_count = (k + TILE_DIM - 1) / TILE_DIM;

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

__global__ void matmul_kernel(float *output, const float *input1,
                              const float *input2, int m, int n, int k,
                              int broadcast) {
    __shared__ float block1[BM][BK];
    __shared__ float block2[BK][BN];

    float thread_output[TM * TN] = {0.0f};
    float reg1[TM];
    float reg2[TN];

    // Calculate matrix offsets
    const int batch_idx = blockIdx.z;
    if (broadcast != 1) {
        input1 += batch_idx * m * k;
    }
    if (broadcast != 2) {
        input2 += batch_idx * k * n;
    }
    output += batch_idx * m * n;

    // Block offsets from matrix
    const int bx = blockIdx.y * BM;
    const int by = blockIdx.x * BN;

    // Thread offsets from block
    const int tx = (threadIdx.x / (BN / TN)) * TM;
    const int ty = (threadIdx.x % (BN / TN)) * TN;

    // Number of threads per block
    const int nthreads = blockDim.x;

    // Strides for tile traversal
    const int win1_stride = nthreads / BK;
    const int win2_stride = nthreads / BN;

    // Coordinates within tiles
    const int x_win1 = threadIdx.x / BK;
    const int y_win1 = threadIdx.x % BK;
    const int x_win2 = threadIdx.x / BN;
    const int y_win2 = threadIdx.x % BN;

    for (int block_offset = 0; block_offset < k; block_offset += BK) {
        int x_block, y_block, x_input, y_input;

        // Load block1 from global memory
        for (int win_offset = 0; win_offset < BM; win_offset += win1_stride) {
            x_block = x_win1 + win_offset;
            y_block = y_win1;
            x_input = x_block + bx;
            y_input = y_block + block_offset;
            block1[x_block][y_block] = (x_input < m && y_input < k)
                                           ? input1[x_input * k + y_input]
                                           : 0.0f;
        }

        // Load block2 from global memory
        for (int win_offset = 0; win_offset < BK; win_offset += win2_stride) {
            x_block = x_win2 + win_offset;
            y_block = y_win2;
            x_input = x_block + block_offset;
            y_input = y_block + by;
            block2[x_block][y_block] = (x_input < k && y_input < n)
                                           ? input2[x_input * n + y_input]
                                           : 0.0f;
        }
        __syncthreads();

        // Compute local tile products and accumulate
        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < TM; j++) {
                reg1[j] = block1[tx + j][i];
            }
            for (int l = 0; l < TN; l++) {
                reg2[l] = block2[i][ty + l];
            }
            for (int j = 0; j < TM; j++) {
                for (int l = 0; l < TN; l++) {
                    thread_output[j * TN + l] += reg1[j] * reg2[l];
                }
            }
        }
        __syncthreads();
    }

    // Write final output back to global memory
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++) {
            int x = bx + tx + j;
            int y = by + ty + l;
            if (x < m && y < n) {
                output[x * n + y] = thread_output[j * TN + l];
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
    float *output_raw = RAW_PTR(output->get_vec());
    const float *input1_raw = RAW_PTR(input1->get_vec());
    const float *input2_raw = RAW_PTR(input2->get_vec());

    dim3 grid_dim(utils::div_ceil(n, BN), utils::div_ceil(m, BM), batch_size);
    dim3 block_dim((BM * BN) / (TM * TN));

    matmul_kernel<<<grid_dim, block_dim>>>(output_raw, input1_raw, input2_raw,
                                           m, n, k, broadcast);
    CUDA_POST_KERNEL_CHECK;
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