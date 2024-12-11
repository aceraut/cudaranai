#include "common.cuh"
#include "linear.cuh"

#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace nnv2 {

// The forward phase of the linear layer involves computing the output as a
// weighted sum of the input features plus a bias term. Given an input X with
// dimensions [batch_size, F_i], the layer computes the output Y as:
//     Y = X * W + b
// where W is the weight matrix with dimensions [F_i, F_o], and b is the bias
// vector with dimensions [1, F_o].

void linear_forward(Array *output, const Array *input, const Array *filter) {
  const ShapeType &output_shape = output->get_shape();
  const ShapeType &input_shape = input->get_shape();
  const ShapeType &filter_shape = filter->get_shape();

  CHECK_EQ(
      input_shape[1],
      filter_shape[0],
      "linear_forward: shape mismatch betwen input and filter");
  CHECK_EQ(
      output_shape[0],
      input_shape[0],
      "linear_forward: shape mismatch between input and output");
  CHECK_EQ(
      output_shape[1],
      filter_shape[1],
      "linear forward: shape mismatch between filter and output");

  ops::matmul(output, input, filter);
}

__global__ void linear_forward_bias_kernel(
    int size,
    float *output,
    const float *bias,
    int out_feats) {
  CUDA_GRID_STRIDE_LOOP(idx, size) {
    int bias_idx = idx % out_feats;
    output[idx] += bias[bias_idx];
  }
}

void linear_forward_bias(Array *output, const Array *bias) {
  const ShapeType &output_shape = output->get_shape();
  const ShapeType &bias_shape = bias->get_shape();

  int batch_size = output_shape[0];
  int out_feats = output_shape[1];

  CHECK_EQ(bias_shape[0], 1, "linear_forward_bias: bias isn't a column vector");
  CHECK_EQ(
      bias_shape[1],
      out_feats,
      "linear_forward_bias: mismatch between bias size and number of "
      "output features");

  int size = batch_size * out_feats;
  int grid_size = utils::div_ceil(size, BLOCK_SIZE);

  float *output_raw = RAW_PTR(output->get_vec());
  const float *bias_raw = RAW_PTR(bias->get_vec());

  linear_forward_bias_kernel<<<grid_size, BLOCK_SIZE>>>(
      size, output_raw, bias_raw, out_feats);
  CUDA_POST_KERNEL_CHECK;
}

// The backward phase of the linear layer involves calculating the loss
// gradients with respect to the input, weight matrix, and bias. Given the
// forward phase equation Y = X * W^T + b, the gradients for the backward phase
// are computed as:
//
// - Loss gradient w.r.t weight matrix: dL/dW = X^T * dL/dY
// - Loss gradient w.r.t input: dL/dX = dL/dY * W^T
// - Loss gradient w.r.t bias: dL/db = sum(dL/dY) along the batch dimension
//
// Here, dL/dY is the gradient of the loss with respect to the output Y, and
// the matrix multiplications and summations are performed according to the
// respective dimensions to compute the gradients.

void linear_backward(
    Array *input_grad,
    Array *filter_grad,
    const Array *input,
    const Array *filter,
    const Array *output_grad,
    ArrayMap &cache) {
  CHECK_EQ(
      filter_grad->get_shape(),
      filter->get_shape(),
      "linear_backward: shape mismatch between filter and its grad");
  CHECK_EQ(
      input_grad->get_shape(),
      input->get_shape(),
      "linear backward: shape mismatch between input and its grad");

  // X^T
  utils::set_array_cache(
      cache, "input_t", {input->get_shape()[1], input->get_shape()[0]});
  ops::transpose(cache["input_t"].get(), input);

  // W^T
  utils::set_array_cache(
      cache, "filter_t", {filter->get_shape()[1], filter->get_shape()[0]});
  ops::transpose(cache["filter_t"].get(), filter);

  ops::matmul(filter_grad, cache["input_t"].get(), output_grad);
  ops::matmul(input_grad, output_grad, cache["filter_t"].get());
}

void linear_backward_bias(Array *bias_grad, const Array *output_grad) {
  const ShapeType &output_grad_shape = output_grad->get_shape();
  const ShapeType &bias_grad_shape = bias_grad->get_shape();

  CHECK_EQ(
      bias_grad_shape[0],
      1,
      "linear_backward_bias: bias grad isn't a column vector");
  CHECK_EQ(
      bias_grad_shape[1],
      output_grad_shape[1],
      "linear_backward_bias: mismatch between bias size and number of "
      "output features");

  ops::sum(bias_grad, output_grad, 0, false);
}

Linear::Linear(int in_feats, int out_feats, const Initializer *init)
    : in_feats(in_feats), out_feats(out_feats) {
  filter.reset(new Array({in_feats, out_feats}));
  bias.reset(new Array({1, out_feats}));
  filter_grad.reset(new Array({in_feats, out_feats}));
  bias_grad.reset(new Array({1, out_feats}));

  // Initialize parameters
  init->initialize(filter.get(), in_feats, out_feats);
  init->initialize(bias.get(), in_feats, out_feats);
}

std::vector<Param> Linear::get_parameters() {
  return {
      std::make_pair(filter.get(), filter_grad.get()),
      std::make_pair(bias.get(), bias_grad.get())};
}

void Linear::forward() {
  const Array *input = prev->get_output();
  int batch_size = input->get_shape()[0];

  utils::set_array_ptr(output, {batch_size, out_feats});

  linear_forward(output.get(), input, filter.get());
  linear_forward_bias(output.get(), bias.get());
}

void Linear::backward() {
  const Array *input = prev->get_output();
  const Array *output_grad = next->get_grad();

  utils::set_array_ptr(grad, input->get_shape());

  linear_backward_bias(bias_grad.get(), output_grad);
  linear_backward(
      grad.get(), filter_grad.get(), input, filter.get(), output_grad, cache);
}

} // namespace nnv2