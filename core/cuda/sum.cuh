#pragma once

#include "../common.cuh"

namespace nnv2 {
namespace cuda {

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

void sum(
    float *output,
    const float *input,
    int output_size,
    int axis_size,
    int stride) {
  int grid_size = utils::div_ceil(output_size, BLOCK_SIZE);

  sum_kernel<<<grid_size, BLOCK_SIZE>>>(
      output_size, output, input, axis_size, stride);
  CUDA_POST_KERNEL_CHECK;
}

} // namespace cuda
} // namespace nnv2