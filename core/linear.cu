// This file implements the Linear or fully connected layer
//
// In this layer, each neuron applies linear transformation to all data points
// in the input vector using its weight. The output is then added with bias.

#include "linear.h"
#include "common.h"

#include <utility>
#include <vector>

namespace nnv2 {

void linear_forward(Array *output, const Array *input, const Array *weight) {
    CHECK_EQ(output->get_shape()[0], input->get_shape()[0],
             "linear_forward: batch size mismatched");
    CHECK_EQ(input->get_shape()[1], weight->get_shape()[0],
             "linear forward: weight row size isn't equal to number of input "
             "features");
    CHECK_EQ(output->get_shape()[1], weight->get_shape()[1],
             "linear forward: weight column size isn't equal to number of "
             "output features");

    func_matmul(output, input, weight);
}

void linear_forward_bias(Array *output, const Array *bias) {
    CHECK_EQ(bias->get_shape()[0], 1,
             "linear_forward_bias: bias isn't a column vector");

    int batch_size = output->get_shape()[0];
    int out_feats = output->get_shape()[1];
    CHECK_EQ(bias->get_shape()[1], out_feats,
             "linear_forward_bias: bias size "
             "isn't equal to number of output features");

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_feats; j++) {
            output->get_vec()[i * out_feats + j] += bias->get_vec()[j];
        }
    }
}

void linear_backward(Array *input_grad, Array *weight_grad, const Array *input,
                     const Array *weight, const Array *output_grad,
                     ArrayMap &cache) {
    CHECK_EQ(weight_grad->get_shape(), weight->get_shape(),
             "linear_backward: weight grad shape isn't equal to weight shape");
    CHECK_EQ(input_grad->get_shape(), input->get_shape(),
             "linear backward: input grad shapee isn't equal to input shape");

    // X^T
    set_array_cache(cache, "input_t",
                    {input->get_shape()[1], input->get_shape()[0]});
    func_transpose(cache["input_t"].get(), input);

    // W^T
    set_array_cache(cache, "weight_t",
                    {weight->get_shape()[1], weight->get_shape()[0]});
    func_transpose(cache["weight_t"].get(), weight);

    // dW = X^T * dA
    func_matmul(weight_grad, cache["input_t"].get(), output_grad);
    // dX = dA * W^T
    func_matmul(input_grad, output_grad, cache["weight_t"].get());
}

void linear_backward_bias(Array *bias_grad, const Array *output_grad) {
    CHECK_EQ(bias_grad->get_shape()[0], 1,
             "linear_backward_bias: bias grad isn't a column vector");
    CHECK_EQ(bias_grad->get_shape()[1], output_grad->get_shape()[1],
             "linear_backward_bias: bias grad size isn't equal to number of "
             "output features");

    // calculate gradient with respect to bias: db = sum(dA, axis=0)
    func_sum(bias_grad, output_grad, 0, false);
}

Linear::Linear(int in_feats, int out_feats, const Initializer *init)
    : in_feats(in_feats), out_feats(out_feats) {
    weight.reset(new Array({in_feats, out_feats}));
    bias.reset(new Array({1, out_feats}));
    weight_grad.reset(new Array({in_feats, out_feats}));
    bias_grad.reset(new Array({1, out_feats}));

    // initialize parameters
    init->initialize(weight.get(), in_feats, out_feats);
    init->initialize(bias.get(), in_feats, out_feats);
}

std::vector<Param> Linear::get_parameters() {
    return {std::make_pair(weight.get(), weight_grad.get()),
            std::make_pair(bias.get(), bias_grad.get())};
}

void Linear::forward() {
    const Array *input = prev->get_output();
    int batch_size = input->get_shape()[0];

    // initialize storage for output
    set_array_ptr(output, {batch_size, out_feats});

    linear_forward(output.get(), input, weight.get());
    linear_forward_bias(output.get(), bias.get());
}

void Linear::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();

    set_array_ptr(grad, input->get_shape());

    linear_backward_bias(bias_grad.get(), output_grad);
    linear_backward(grad.get(), weight_grad.get(), input, weight.get(),
                    output_grad, cache);
}

} // namespace nnv2