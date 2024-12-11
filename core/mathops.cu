#include "common.cuh"
#include "cuda/arithmetic.cuh"
#include "cuda/matmul.cuh"
#include "cuda/mean.cuh"
#include "cuda/sum.cuh"
#include "cuda/transpose.cuh"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

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

  cuda::add(output_vec, input1_vec, input2_vec);
}

void add(Array *output, const Array *input, float value) {
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::add: size mismatch between input and output");

  cuda::add(output_vec, input_vec, value);
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

  cuda::subtract(output_vec, input1_vec, input2_vec);
}

void subtract(Array *output, const Array *input, float value) {
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::subtract: size mismatch between input and output");

  cuda::subtract(output_vec, input_vec, value);
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

  cuda::multiply(output_vec, input1_vec, input2_vec);
}

void multiply(Array *output, const Array *input, float value) {
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::multiply: size mismatch between input and output");

  cuda::multiply(output_vec, input_vec, value);
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

  cuda::divide(output_vec, input1_vec, input2_vec);
}

void log(Array *output, const Array *input) {
  const VecType &input_vec = input->get_vec();
  VecType &output_vec = output->get_vec();

  CHECK_EQ(
      output_vec.size(),
      input_vec.size(),
      "ops::log: size mismatch between input and output");

  cuda::log(output_vec, input_vec);
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

  cuda::matmul(
      output_raw, input1_raw, input2_raw, batch_size, m, n, k, broadcast);
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

  cuda::transpose(output_raw, input_raw, batch_size, m, n);
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

  cuda::sum(output_raw, input_raw, output_size, axis_size, stride);
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

  cuda::mean(output_raw, input_raw, output_size, axis_size, stride);
}

} // namespace ops
} // namespace nnv2