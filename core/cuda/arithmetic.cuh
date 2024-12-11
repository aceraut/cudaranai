#pragma once

#include "../common.cuh"

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace nnv2 {
namespace cuda {

void add(VecType &output, const VecType &input1, const VecType &input2) {
  thrust::transform(
      input1.begin(),
      input1.end(),
      input2.begin(),
      output.begin(),
      thrust::plus<float>());
}

void add(VecType &output, const VecType &input, float value) {
  thrust::transform(
      input.begin(), input.end(), output.begin(), [value] __device__(float x) {
        return x + value;
      });
}

void subtract(VecType &output, const VecType &input1, const VecType &input2) {
  thrust::transform(
      input1.begin(),
      input1.end(),
      input2.begin(),
      output.begin(),
      thrust::minus<float>());
}

void subtract(VecType &output, const VecType &input, float value) {
  thrust::transform(
      input.begin(), input.end(), output.begin(), [value] __device__(float x) {
        return x - value;
      });
}

void multiply(VecType &output, const VecType &input1, const VecType &input2) {
  thrust::transform(
      input1.begin(),
      input1.end(),
      input2.begin(),
      output.begin(),
      thrust::multiplies<float>());
}

void multiply(VecType &output, const VecType &input, float value) {
  thrust::transform(
      input.begin(), input.end(), output.begin(), [value] __device__(float x) {
        return x * value;
      });
}

void divide(VecType &output, const VecType &input1, const VecType &input2) {
  thrust::transform(
      input1.begin(),
      input1.end(),
      input2.begin(),
      output.begin(),
      thrust::divides<float>());
}

void log(VecType &output, const VecType &input) {
  thrust::transform(
      input.begin(), input.end(), output.begin(), [] __device__(float x) {
        return logf(x);
      });
}

} // namespace cuda
} // namespace nnv2