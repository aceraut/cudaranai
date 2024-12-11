#pragma once

#include "../common.cuh"

#include <cuda_runtime.h>

namespace nnv2 {
namespace cuda {

constexpr int XPOSE_BM = 8;
constexpr int XPOSE_BN = 32;

__global__ void
transpose_kernel(float *output, const float *input, int m, int n) {
  __shared__ float tile[XPOSE_BN][XPOSE_BN + 1];

  int batch_idx = blockIdx.z;
  input += batch_idx * m * n;
  output += batch_idx * n * m;

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

void transpose(
    float *output,
    const float *input,
    int batch_size,
    int m,
    int n) {
  dim3 grid_dim(
      utils::div_ceil(n, XPOSE_BN), utils::div_ceil(m, XPOSE_BN), batch_size);
  dim3 block_dim(XPOSE_BN, XPOSE_BM);

  transpose_kernel<<<grid_dim, block_dim>>>(output, input, m, n);
  CUDA_POST_KERNEL_CHECK;
}

} // namespace cuda
} // namespace nnv2