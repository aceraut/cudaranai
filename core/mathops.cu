#include "common.cuh"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace nnv2 {
namespace ops {

// Addition
void add(Array *output, const Array *input1, const Array *input2) {
  const VecType<float> &input1_vec = input1->get_vec();
  const VecType<float> &input2_vec = input2->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      input1_vec.size(),
      input2_vec.size(),
      "ops::add: size mismatch between inputs");
  CHECK_EQ(
      output_vec.size(),
      input1_vec.size(),
      "ops::add: size mismatched between input and output");

  thrust::transform(
      input1_vec.begin(),
      input1_vec.end(),
      input2_vec.begin(),
      output_vec.begin(),
      thrust::plus<float>());
}

void add(Array *output, const Array *input, float value) {
  const VecType<float> &input_vec = input->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::add: size mismatch between input and output");

  thrust::transform(
      input_vec.begin(),
      input_vec.end(),
      output_vec.begin(),
      [value] __device__(float x) { return x + value; });
}

// Subtraction
void subtract(Array *output, const Array *input1, const Array *input2) {
  const VecType<float> &input1_vec = input1->get_vec();
  const VecType<float> &input2_vec = input2->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      input1_vec.size(),
      input2_vec.size(),
      "ops::subtract: size mismatch between inputs");
  CHECK_EQ(
      output_vec.size(),
      input1_vec.size(),
      "ops::subtract: size mismatch between input and outputs");

  thrust::transform(
      input1_vec.begin(),
      input1_vec.end(),
      input2_vec.begin(),
      output_vec.begin(),
      thrust::minus<float>());
}

void subtract(Array *output, const Array *input, float value) {
  const VecType<float> &input_vec = input->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::subtract: size mismatch between input and output");

  thrust::transform(
      input_vec.begin(),
      input_vec.end(),
      output_vec.begin(),
      [value] __device__(float x) { return x - value; });
}

// Element-wise multiplication
void multiply(Array *output, const Array *input1, const Array *input2) {
  const VecType<float> &input1_vec = input1->get_vec();
  const VecType<float> &input2_vec = input2->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      input1_vec.size(),
      input2_vec.size(),
      "ops::multiply: size mismatch between inputs");
  CHECK_EQ(
      output_vec.size(),
      input1_vec.size(),
      "ops::multiply: size mismatch between input and outputs");

  thrust::transform(
      input1_vec.begin(),
      input1_vec.end(),
      input2_vec.begin(),
      output_vec.begin(),
      thrust::multiplies<float>());
}

void multiply(Array *output, const Array *input, float value) {
  const VecType<float> &input_vec = input->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::multiply: size mismatch between input and output");

  thrust::transform(
      input_vec.begin(),
      input_vec.end(),
      output_vec.begin(),
      [value] __device__(float x) { return x * value; });
}

// Element-wise division
void divide(Array *output, const Array *input1, const Array *input2) {
  const VecType<float> &input1_vec = input1->get_vec();
  const VecType<float> &input2_vec = input2->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      input1_vec.size(),
      input2_vec.size(),
      "ops::divide: size mismatch between inputs");
  CHECK_EQ(
      output_vec.size(),
      input1_vec.size(),
      "ops::divide: size mismatch between input and outputs");

  thrust::transform(
      input1_vec.begin(),
      input1_vec.end(),
      input2_vec.begin(),
      output_vec.begin(),
      thrust::divides<float>());
}

// Logarithm operation
void log(Array *output, const Array *input) {
  const VecType<float> &input_vec = input->get_vec();
  VecType<float> &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::log: size mismatch between input and output");

  thrust::transform(
      input_vec.begin(),
      input_vec.end(),
      output_vec.begin(),
      [] __device__(float x) { return logf(x); });
}

// Matrix multiplication

// Dimension threshold to use the matmul kernel v1
constexpr int MMUL_LIM = 512;
// Block tile dimension in the matmul kernel v1
constexpr int MMUL1_BD = 16;
// Block tile dimension in the matmul kernel v2
constexpr int MMUL2_BM = 64;
constexpr int MMUL2_BN = 64;
constexpr int MMUL2_BK = 8;
// Thread tile dimension in the matmul kernel v2
constexpr int MMUL2_TM = 8;
constexpr int MMUL2_TN = 8;

__global__ void matmul_kernel_v1(
    float *output,
    const float *input1,
    const float *input2,
    int m,
    int n,
    int k,
    int broadcast) {
  __shared__ float tile1[MMUL1_BD][MMUL1_BD];
  __shared__ float tile2[MMUL1_BD][MMUL1_BD];

  // Calculate offsets of the matrices
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

  int x = bx * MMUL1_BD + tx;
  int y = by * MMUL1_BD + ty;

  float thread_output = 0.0;

  for (int block_offset = 0; block_offset < k; block_offset += MMUL1_BD) {
    tile1[tx][ty] = (x < m && ty + block_offset < k)
                        ? input1[x * k + ty + block_offset]
                        : 0.0;
    tile2[tx][ty] = (y < n && tx + block_offset < k)
                        ? input2[(tx + block_offset) * n + y]
                        : 0.0;
    __syncthreads();

    for (int j = 0; j < MMUL1_BD; j++) {
      thread_output += tile1[tx][j] * tile2[j][ty];
    }
    __syncthreads();
  }

  if (x < m && y < n) {
    output[x * n + y] = thread_output;
  }
}

__global__ void matmul_kernel_v2(
    float *output,
    const float *input1,
    const float *input2,
    int m,
    int n,
    int k,
    int broadcast) {
  __shared__ float tile1[MMUL2_BM][MMUL2_BK];
  __shared__ float tile2[MMUL2_BK][MMUL2_BN];

  float thread_output[MMUL2_TM * MMUL2_TN] = {0.0};
  float reg1[MMUL2_TM];
  float reg2[MMUL2_TN];

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
  int bx = blockIdx.y * MMUL2_BM;
  int by = blockIdx.x * MMUL2_BN;

  // Thread offsets from block
  int tx = (threadIdx.x / (MMUL2_BN / MMUL2_TN)) * MMUL2_TM;
  int ty = (threadIdx.x % (MMUL2_BN / MMUL2_TN)) * MMUL2_TN;

  // Number of threads per block
  int nthreads = blockDim.x;

  // Strides for tile traversal
  int win1_stride = nthreads / MMUL2_BK;
  int win2_stride = nthreads / MMUL2_BN;

  // Coordinates within tiles
  int x_win1 = threadIdx.x / MMUL2_BK;
  int y_win1 = threadIdx.x % MMUL2_BK;
  int x_win2 = threadIdx.x / MMUL2_BN;
  int y_win2 = threadIdx.x % MMUL2_BN;

  for (int block_offset = 0; block_offset < k; block_offset += MMUL2_BK) {
    int x_block, y_block, x_input, y_input;

    // Load tile1 from global memory
    for (int win_offset = 0; win_offset < MMUL2_BM; win_offset += win1_stride) {
      x_block = x_win1 + win_offset;
      y_block = y_win1;
      x_input = x_block + bx;
      y_input = y_block + block_offset;
      tile1[x_block][y_block] =
          (x_input < m && y_input < k) ? input1[x_input * k + y_input] : 0.0;
    }

    // Load tile2 from global memory
    for (int win_offset = 0; win_offset < MMUL2_BK; win_offset += win2_stride) {
      x_block = x_win2 + win_offset;
      y_block = y_win2;
      x_input = x_block + block_offset;
      y_input = y_block + by;
      tile2[x_block][y_block] =
          (x_input < k && y_input < n) ? input2[x_input * n + y_input] : 0.0;
    }
    __syncthreads();

    // Compute local tile products and accumulate
    for (int i = 0; i < MMUL2_BK; i++) {
      for (int j = 0; j < MMUL2_TM; j++) {
        reg1[j] = tile1[tx + j][i];
      }
      for (int l = 0; l < MMUL2_TN; l++) {
        reg2[l] = tile2[i][ty + l];
      }
      for (int j = 0; j < MMUL2_TM; j++) {
        for (int l = 0; l < MMUL2_TN; l++) {
          thread_output[j * MMUL2_TN + l] += reg1[j] * reg2[l];
        }
      }
    }
    __syncthreads();
  }

  // Write final output back to global memory
  for (int j = 0; j < MMUL2_TM; j++) {
    for (int l = 0; l < MMUL2_TN; l++) {
      int x = bx + tx + j;
      int y = by + ty + l;
      if (x < m && y < n) {
        output[x * n + y] = thread_output[j * MMUL2_TN + l];
      }
    }
  }
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
void matmul(
    Array *output,
    const Array *input1,
    const Array *input2,
    int broadcast) {
  const ShapeType &output_shape = output->get_shape();
  const ShapeType &input1_shape = input1->get_shape();
  const ShapeType &input2_shape = input2->get_shape();

  CHECK_COND(
      input1_shape.size() > 1, "ops::matmul: shape error at first input");
  CHECK_COND(
      input2_shape.size() > 1, "ops::matmul: shape error at second input");
  CHECK_COND(output_shape.size() > 1, "ops::matmul: shape error at output");

  // Additional dimension check for broadcast case
  if (broadcast == 1) {
    CHECK_EQ(input1_shape.size(), 2, "ops::matmul: shape error at first input");
  } else if (broadcast == 2) {
    CHECK_EQ(
        input2_shape.size(), 2, "ops::matmul: shape error at second input");
  }

  // Calculate and validate batch size
  int batch_size = std::accumulate(
      output_shape.begin(), output_shape.end() - 2, 1, std::multiplies<int>());
  int bs_input1 = std::accumulate(
      input1_shape.begin(), input1_shape.end() - 2, 1, std::multiplies<int>());
  int bs_input2 = std::accumulate(
      input2_shape.begin(), input2_shape.end() - 2, 1, std::multiplies<int>());

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
  CHECK_EQ(
      m,
      output_h,
      "ops::matmul: shape mismatch between first input and output");
  CHECK_EQ(
      n,
      output_w,
      "ops::matmul: shape mismatch between second input and output");

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input1_raw = RAW_PTR(input1->get_vec());
  const float *input2_raw = RAW_PTR(input2->get_vec());

  // Select which kernel to launch based on the dimension.
  // Kernel v1 works better on small to medium matrices, while v2 works better
  // on large matrices.
  if (m <= MMUL_LIM && n <= MMUL_LIM && k <= MMUL_LIM) {
    dim3 grid_dim(
        utils::div_ceil(n, MMUL1_BD), utils::div_ceil(m, MMUL1_BD), batch_size);
    dim3 block_dim(MMUL1_BD, MMUL1_BD);

    matmul_kernel_v1<<<grid_dim, block_dim>>>(
        output_raw, input1_raw, input2_raw, m, n, k, broadcast);
  } else {
    dim3 grid_dim(
        utils::div_ceil(n, MMUL2_BN), utils::div_ceil(m, MMUL2_BM), batch_size);
    dim3 block_dim((MMUL2_BM * MMUL2_BN) / (MMUL2_TM * MMUL2_TN));

    matmul_kernel_v2<<<grid_dim, block_dim>>>(
        output_raw, input1_raw, input2_raw, m, n, k, broadcast);
  }
  CUDA_POST_KERNEL_CHECK;
}

// Transpose operation
constexpr int XPOSE_BM = 8;
constexpr int XPOSE_BN = 32;

__global__ void
transpose_kernel(float *output, const float *input, int m, int n) {
  __shared__ float tile[XPOSE_BN][XPOSE_BN + 1];

  int batch_idx = blockIdx.z;
  input += batch_idx * m * n;
  output += batch_idx * n * m;

  // Diagonal block reordering
  // https://www.csd.uwo.ca/~mmorenom/HPC-Slides/Optimizing_CUDA_Code-2x2.pdf
  int bx, by;
  if (m == n) {
    bx = blockIdx.x;
    by = (blockIdx.x + blockIdx.y) % gridDim.x;
  } else {
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    bx = bid % gridDim.y;
    by = ((bid / gridDim.y) + bx) % gridDim.x;
  }

  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int x = bx * XPOSE_BN + tx;
  int y = by * XPOSE_BN + ty;

  // A block of threads load a grid of XPOSE_BM rows and XPOSE_BN columns into
  // shared memory in each iteration.
  for (int i = 0; i < XPOSE_BN; i += XPOSE_BM) {
    if (x + i < m && y < n) {
      tile[tx + i][ty] = input[(x + i) * n + y];
    }
  }
  __syncthreads();

  x = by * XPOSE_BN + tx;
  y = bx * XPOSE_BN + ty;

  for (int i = 0; i < XPOSE_BN; i += XPOSE_BM) {
    if (x + i < n && y < m) {
      output[(x + i) * m + y] = tile[ty][tx + i];
    }
  }
}

// Performs matrix tranpose. If the input has more than 2 dimensions, batch
// matrix transpose is performed, which requires output to have the same batch
// size as the input array
void transpose(Array *output, const Array *input) {
  const ShapeType &output_shape = output->get_shape();
  const ShapeType &input_shape = input->get_shape();

  // Check if the dimensions are at least 2
  CHECK_COND(input_shape.size() > 1, "ops::transpose: shape error at input");
  CHECK_COND(output_shape.size() > 1, "ops::transpose: shape error at output");

  // Calculate batch size and validate
  int batch_size = std::accumulate(
      output_shape.begin(), output_shape.end() - 2, 1, std::multiplies<int>());
  int bs_input = std::accumulate(
      input_shape.begin(), input_shape.end() - 2, 1, std::multiplies<int>());
  CHECK_EQ(batch_size, bs_input, "ops::transpose: batch size mismatch");

  // Validate matrix dimension
  int m = *(input_shape.rbegin() + 1);
  int n = *(input_shape.rbegin());
  int output_h = *(output_shape.rbegin() + 1);
  int output_w = *(output_shape.rbegin());

  CHECK_EQ(
      m, output_w, "ops::transpose: shape mismatch between input and output");
  CHECK_EQ(
      n, output_h, "ops::transpose: shape mismatch between input and output");

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  dim3 grid_dim(
      utils::div_ceil(n, XPOSE_BN), utils::div_ceil(m, XPOSE_BN), batch_size);
  dim3 block_dim(XPOSE_BN, XPOSE_BM);

  transpose_kernel<<<grid_dim, block_dim>>>(output_raw, input_raw, m, n);
  CUDA_POST_KERNEL_CHECK;
}

// Sum reduction over a dimension (or axis)
__global__ void sum_kernel(
    int size,
    float *output,
    const float *input,
    int axis_size,
    int stride) {
  CUDA_GRID_STRIDE_LOOP(idx, size) {
    int base = (idx / stride) * axis_size * stride + (idx % stride);

    float value = 0;
    for (int i = 0; i < axis_size; i++) {
      value += input[base + i * stride];
    }
    output[idx] = value;
  }
}

// Calculates sum of array elements along a given axis. The parameter `reduce`
// indicates whether the dimension at `axis` in input array is removed in the
// output.
void sum(Array *output, const Array *input, int axis, bool reduce) {
  const ShapeType &input_shape = input->get_shape();
  const ShapeType &output_shape = output->get_shape();

  CHECK_COND(
      axis >= 0, "ops::sum: support for negative axis isn't implemented");
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

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  int output_size = output->get_vec().size();
  int axis_size = input_shape[axis];
  int stride = std::accumulate(
      input_shape.begin() + axis + 1,
      input_shape.end(),
      1,
      std::multiplies<int>());

  int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

  sum_kernel<<<grid_size, BLOCK_SIZE>>>(
      output_size, output_raw, input_raw, axis_size, stride);
  CUDA_POST_KERNEL_CHECK;
}

// Mean reduction over a dimension (or axis)
__global__ void mean_kernel(
    int size,
    float *output,
    const float *input,
    int axis_size,
    int stride) {
  CUDA_GRID_STRIDE_LOOP(idx, size) {
    int base = (idx / stride) * axis_size * stride + (idx % stride);

    float value = 0;
    for (int i = 0; i < axis_size; i++) {
      value += input[base + i * stride];
    }
    output[idx] = value / axis_size;
  }
}

// Calculates mean value of array elements along a given axis. The parameter
// `reduce` indicates whether the dimension at `axis` in input array is removed
// in the output.
void mean(Array *output, const Array *input, int axis, bool reduce) {
  const ShapeType &input_shape = input->get_shape();
  const ShapeType &output_shape = output->get_shape();

  CHECK_COND(
      axis >= 0, "ops::mean: support for negative axis isn't implemented");
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

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  int output_size = output->get_vec().size();
  int axis_size = input_shape[axis];
  int stride = std::accumulate(
      input_shape.begin() + axis + 1,
      input_shape.end(),
      1,
      std::multiplies<int>());

  int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

  mean_kernel<<<grid_size, BLOCK_SIZE>>>(
      output_size, output_raw, input_raw, axis_size, stride);
  CUDA_POST_KERNEL_CHECK;
}

} // namespace ops
} // namespace nnv2