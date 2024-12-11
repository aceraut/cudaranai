#pragma once

#include "../common.cuh"

#include <cuda_runtime.h>

namespace nnv2 {
namespace cuda {

constexpr int MMUL1_TD = 16;

__global__ void matmul_kernel_v1(
    float *output,
    const float *input1,
    const float *input2,
    int m,
    int n,
    int k,
    int broadcast) {
  __shared__ float tile1[MMUL1_TD][MMUL1_TD];
  __shared__ float tile2[MMUL1_TD][MMUL1_TD];

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

  int x = bx * MMUL1_TD + tx;
  int y = by * MMUL1_TD + ty;

  // Loop over input tiles to calculate the dot value
  float thread_output = 0.0;
  int tile_count = (k + MMUL1_TD - 1) / MMUL1_TD;

  for (int i = 0; i < tile_count; i++) {
    tile1[tx][ty] = (x < m && i * MMUL1_TD + ty < k)
                        ? input1[x * k + i * MMUL1_TD + ty]
                        : 0.0;

    tile2[tx][ty] = (y < n && i * MMUL1_TD + tx < k)
                        ? input2[(i * MMUL1_TD + tx) * n + y]
                        : 0.0;
    __syncthreads();

    for (int j = 0; j < MMUL1_TD; j++) {
      thread_output += tile1[tx][j] * tile2[j][ty];
    }
    __syncthreads();
  }

  if (x < m && y < n) {
    output[x * n + y] = thread_output;
  }
}

constexpr int MMUL2_BM = 64;
constexpr int MMUL2_BN = 64;
constexpr int MMUL2_BK = 8;
constexpr int MMUL2_TM = 8;
constexpr int MMUL2_TN = 8;

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

constexpr int MMUL1_LIM = 512;

void matmul(
    float *output,
    const float *input1,
    const float *input2,
    int batch_size,
    int m,
    int n,
    int k,
    int broadcast) {
  if (m <= MMUL1_LIM && n <= MMUL1_LIM && k <= MMUL1_LIM) {
  //if (true) {
    dim3 grid_dim(
        utils::div_ceil(n, MMUL1_TD), utils::div_ceil(m, MMUL1_TD), batch_size);
    dim3 block_dim(MMUL1_TD, MMUL1_TD);

    matmul_kernel_v1<<<grid_dim, block_dim>>>(
        output, input1, input2, m, n, k, broadcast);
  } else {
    dim3 grid_dim(
        utils::div_ceil(n, MMUL2_BN), utils::div_ceil(m, MMUL2_BM), batch_size);
    dim3 block_dim((MMUL2_BM * MMUL2_BN) / (MMUL2_TM * MMUL2_TN));

    matmul_kernel_v2<<<grid_dim, block_dim>>>(
        output, input1, input2, m, n, k, broadcast);
  }

  CUDA_POST_KERNEL_CHECK;
}

} // namespace cuda
} // namespace nnv2