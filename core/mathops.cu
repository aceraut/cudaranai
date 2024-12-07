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

void add(Array *output, const Array *input1, const Array *input2) {
  const VecType &input1_vec = input1->get_vec();
  const VecType &input2_vec = input2->get_vec();
  VecType &output_vec = output->get_vec();

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
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

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

void subtract(Array *output, const Array *input1, const Array *input2) {
  const VecType &input1_vec = input1->get_vec();
  const VecType &input2_vec = input2->get_vec();
  VecType &output_vec = output->get_vec();

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
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

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

void multiply(Array *output, const Array *input1, const Array *input2) {
  const VecType &input1_vec = input1->get_vec();
  const VecType &input2_vec = input2->get_vec();
  VecType &output_vec = output->get_vec();

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
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

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

void divide(Array *output, const Array *input1, const Array *input2) {
  const VecType &input1_vec = input1->get_vec();
  const VecType &input2_vec = input2->get_vec();
  VecType &output_vec = output->get_vec();

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

void log(Array *output, const Array *input) {
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::log: size mismatch between input and output");

  thrust::transform(
      input_vec.begin(),
      input_vec.end(),
      output_vec.begin(),
      [] __device__(float e) { return logf(e); });
}

// Matrix multiplication helper kernel
__global__ void matmul_kernel(
    float *output,
    const float *input1,
    const float *input2,
    int m,
    int n,
    int k,
    int broadcast) {
  __shared__ float tile1[MMUL_BM][MMUL_BK];
  __shared__ float tile2[MMUL_BK][MMUL_BN];

  float thread_output[MMUL_TM * MMUL_TN] = {0.0f};
  float reg1[MMUL_TM];
  float reg2[MMUL_TN];

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
  int bx = blockIdx.y * MMUL_BM;
  int by = blockIdx.x * MMUL_BN;

  // Thread offsets from block
  int tx = (threadIdx.x / (MMUL_BN / MMUL_TN)) * MMUL_TM;
  int ty = (threadIdx.x % (MMUL_BN / MMUL_TN)) * MMUL_TN;

  // Number of threads per block
  int nthreads = blockDim.x;

  // Strides for tile traversal
  int win1_stride = nthreads / MMUL_BK;
  int win2_stride = nthreads / MMUL_BN;

  // Coordinates within tiles
  int x_win1 = threadIdx.x / MMUL_BK;
  int y_win1 = threadIdx.x % MMUL_BK;
  int x_win2 = threadIdx.x / MMUL_BN;
  int y_win2 = threadIdx.x % MMUL_BN;

  for (int block_offset = 0; block_offset < k; block_offset += MMUL_BK) {
    int x_block, y_block, x_input, y_input;

    // Load tile1 from global memory
    for (int win_offset = 0; win_offset < MMUL_BM; win_offset += win1_stride) {
      x_block = x_win1 + win_offset;
      y_block = y_win1;
      x_input = x_block + bx;
      y_input = y_block + block_offset;
      tile1[x_block][y_block] =
          (x_input < m && y_input < k) ? input1[x_input * k + y_input] : 0.0f;
    }

    // Load tile2 from global memory
    for (int win_offset = 0; win_offset < MMUL_BK; win_offset += win2_stride) {
      x_block = x_win2 + win_offset;
      y_block = y_win2;
      x_input = x_block + block_offset;
      y_input = y_block + by;
      tile2[x_block][y_block] =
          (x_input < k && y_input < n) ? input2[x_input * n + y_input] : 0.0f;
    }
    __syncthreads();

    // Compute local tile products and accumulate
    for (int i = 0; i < MMUL_BK; i++) {
      for (int j = 0; j < MMUL_TM; j++) {
        reg1[j] = tile1[tx + j][i];
      }
      for (int l = 0; l < MMUL_TN; l++) {
        reg2[l] = tile2[i][ty + l];
      }
      for (int j = 0; j < MMUL_TM; j++) {
        for (int l = 0; l < MMUL_TN; l++) {
          thread_output[j * MMUL_TN + l] += reg1[j] * reg2[l];
        }
      }
    }
    __syncthreads();
  }

  // Write final output back to global memory
  for (int j = 0; j < MMUL_TM; j++) {
    for (int l = 0; l < MMUL_TN; l++) {
      int x = bx + tx + j;
      int y = by + ty + l;
      if (x < m && y < n) {
        output[x * n + y] = thread_output[j * MMUL_TN + l];
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
    Array *output, const Array *input1, const Array *input2, int broadcast) {
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

  // Launch kernel
  float *output_raw = RAW_PTR(output->get_vec());
  const float *input1_raw = RAW_PTR(input1->get_vec());
  const float *input2_raw = RAW_PTR(input2->get_vec());

  dim3 grid_dim(
      utils::div_ceil(n, MMUL_BN), utils::div_ceil(m, MMUL_BM), batch_size);
  dim3 block_dim((MMUL_BM * MMUL_BN) / (MMUL_TM * MMUL_TN));

  matmul_kernel<<<grid_dim, block_dim>>>(
      output_raw, input1_raw, input2_raw, m, n, k, broadcast);
  CUDA_POST_KERNEL_CHECK;
}

// Matrix transpose helper kernel
__global__ void
transpose_kernel(float *output, const float *input, int m, int n) {
  __shared__ float tile[XPOSE_BN][XPOSE_BN];

  int batch_idx = blockIdx.z;
  input += batch_idx * m * n;
  output += batch_idx * n * m;

  int bx = blockIdx.y;
  int by = blockIdx.x;
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int x = bx * XPOSE_BN + tx;
  int y = by * XPOSE_BN + ty;

  if (x < m && y < n) {
    tile[tx][ty] = input[x * n + y];
    __syncthreads();
    output[y * m + x] = tile[tx][ty];
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

  // Launch kernels
  dim3 grid_dim(
      utils::div_ceil(n, XPOSE_BN), utils::div_ceil(m, XPOSE_BN), batch_size);
  dim3 block_dim(XPOSE_BN, XPOSE_BN);

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  transpose_kernel<<<grid_dim, block_dim>>>(output_raw, input_raw, m, n);
  CUDA_POST_KERNEL_CHECK;
}

// Axis sum helper kernel
__global__ void sum_kernel(
    int size, float *output, const float *input, int axis_size, int stride) {
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

  // Launch kernels
  int output_size = output->get_vec().size();
  int axis_size = input_shape[axis];
  int stride = std::accumulate(
      input_shape.begin() + axis + 1,
      input_shape.end(),
      1,
      std::multiplies<int>());
  int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  sum_kernel<<<grid_size, BLOCK_SIZE>>>(
      output_size, output_raw, input_raw, axis_size, stride);
  CUDA_POST_KERNEL_CHECK;
}

// Axis mean helper kernel
__global__ void mean_kernel(
    int size, float *output, const float *input, int axis_size, int stride) {
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

  // Launch kernels
  int output_size = output->get_vec().size();
  int axis_size = input_shape[axis];
  int stride = std::accumulate(
      input_shape.begin() + axis + 1,
      input_shape.end(),
      1,
      std::multiplies<int>());
  int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

  float *output_raw = RAW_PTR(output->get_vec());
  const float *input_raw = RAW_PTR(input->get_vec());

  mean_kernel<<<grid_size, BLOCK_SIZE>>>(
      output_size, output_raw, input_raw, axis_size, stride);
  CUDA_POST_KERNEL_CHECK;
}

} // namespace ops
} // namespace nnv2