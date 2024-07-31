// This file implements the Linear or fully connected layer
//
// In this layer, each neuron applies linear transformation to all data points
// in the input vector using its filter. The output is then added with bias.

#include "common.cuh"
#include "linear.cuh"

#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace nnv2 {

void linear_forward(Array *output, const Array *input, const Array *filter) {
    CHECK_EQ(input->get_shape()[1], filter->get_shape()[0],
             "linear_forward: shape mismatch betwen input and filter");
    CHECK_EQ(output->get_shape()[0], input->get_shape()[0],
             "linear_forward: shape mismatch between input and output");
    CHECK_EQ(output->get_shape()[1], filter->get_shape()[1],
             "linear forward: shape mismatch between filter and output");

    func_matmul(output, input, filter);
}

__global__ void linear_forward_bias_kernel(int size, float *output,
                                           const float *bias, int out_feats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % out_feats;
        output[idx] += bias[bias_idx];
    }
}

void linear_forward_bias(Array *output, const Array *bias) {
    CHECK_EQ(bias->get_shape()[0], 1,
             "linear_forward_bias: bias isn't a column vector");

    int batch_size = output->get_shape()[0];
    int out_feats = output->get_shape()[1];
    CHECK_EQ(bias->get_shape()[1], out_feats,
             "linear_forward_bias: mismatch between bias size and number of "
             "output features");

    float *output_raw = RAW_PTR(output->get_vec());
    const float *bias_raw = RAW_PTR(bias->get_vec());

    int size = batch_size * out_feats;
    int grid_size = ceil(float(size) / BLOCK_SIZE);

    linear_forward_bias_kernel<<<grid_size, BLOCK_SIZE>>>(size, output_raw,
                                                          bias_raw, out_feats);
    CUDA_POST_KERNEL_CHECK;
}

void linear_backward(Array *input_grad, Array *filter_grad, const Array *input,
                     const Array *filter, const Array *output_grad,
                     ArrayMap &cache) {
    CHECK_EQ(filter_grad->get_shape(), filter->get_shape(),
             "linear_backward: shape mismatch between filter and its grad");
    CHECK_EQ(input_grad->get_shape(), input->get_shape(),
             "linear backward: shape mismatch between input and its grad");

    // X^T
    set_array_cache(cache, "input_t",
                    {input->get_shape()[1], input->get_shape()[0]});
    func_transpose(cache["input_t"].get(), input);

    // W^T
    set_array_cache(cache, "filter_t",
                    {filter->get_shape()[1], filter->get_shape()[0]});
    func_transpose(cache["filter_t"].get(), filter);

    // dW = X^T * dA
    func_matmul(filter_grad, cache["input_t"].get(), output_grad);
    // dX = dA * W^T
    func_matmul(input_grad, output_grad, cache["filter_t"].get());
}

void linear_backward_bias(Array *bias_grad, const Array *output_grad) {
    CHECK_EQ(bias_grad->get_shape()[0], 1,
             "linear_backward_bias: bias grad isn't a column vector");
    CHECK_EQ(bias_grad->get_shape()[1], output_grad->get_shape()[1],
             "linear_backward_bias: mismatch between bias size and number of "
             "output features");

    // calculate gradient with respect to bias: db = sum(dA, axis=0)
    func_sum(bias_grad, output_grad, 0, false);
}

Linear::Linear(int in_feats, int out_feats, const Initializer *init)
    : in_feats(in_feats), out_feats(out_feats) {
    filter.reset(new Array({in_feats, out_feats}));
    bias.reset(new Array({1, out_feats}));
    filter_grad.reset(new Array({in_feats, out_feats}));
    bias_grad.reset(new Array({1, out_feats}));

    // initialize parameters
    init->initialize(filter.get(), in_feats, out_feats);
    init->initialize(bias.get(), in_feats, out_feats);
}

std::vector<Param> Linear::get_parameters() {
    return {std::make_pair(filter.get(), filter_grad.get()),
            std::make_pair(bias.get(), bias_grad.get())};
}

void Linear::forward() {
    const Array *input = prev->get_output();
    int batch_size = input->get_shape()[0];

    // initialize storage for output
    set_array_ptr(output, {batch_size, out_feats});

    linear_forward(output.get(), input, filter.get());
    linear_forward_bias(output.get(), bias.get());
}

void Linear::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();

    set_array_ptr(grad, input->get_shape());

    linear_backward_bias(bias_grad.get(), output_grad);
    linear_backward(grad.get(), filter_grad.get(), input, filter.get(),
                    output_grad, cache);
}

} // namespace nnv2