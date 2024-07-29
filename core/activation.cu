// This file implements multiple types of activation layers

#include "activation.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nnv2 {

// ReLU
//
// ReLU activation: R(x) = max(0, x)
// ReLU derivative: dR/dx = (x > 0) ? 0 : 1
//
// Note that this is a drop-in layer that directly alters previous layer's
// output and next layer's input gradient
void relu_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "relu_forward: output size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output->get_vec().begin(),
                   [](float x) { return fmaxf(0.0, x); });
}

void relu_backward(Array *input_grad, const Array *output_grad,
                   const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "relu_backward: input grad size isn't equal to output grad size");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "relu_backward: input grad size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output_grad->get_vec().begin(),
                   input_grad->get_vec().begin(),
                   [](float x, float g) { return x > EPS ? g : 0; });
}

void ReLU::forward() {
    Array *input = prev->get_output();
    relu_forward(input, input);
}

void ReLU::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    relu_backward(output_grad, output_grad, input);
}

// Sigmoid
//
// Sigmoid activation: S(x) = 1 / (1 + e^(-x))
// Sigmoid derivative: dS/dx = (1 - S(x)) * S(x)
//
// Note that this is a drop-in layer that directly alters previous layer's
// output and next layer's input gradient
void sigmoid_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "sigmoid_forward: output size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output->get_vec().begin(),
                   [](float x) { return 1 / (1 + expf(-x)); });
}

void sigmoid_backward(Array *input_grad, const Array *output_grad,
                      const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "sigmoid_backward: input grad size isn't equal to output grad "
             "size");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "sigmoid_backward: input grad size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output_grad->get_vec().begin(),
                   input_grad->get_vec().begin(), [](float x, float g) {
                       float sigmoid = 1 / (1 + expf(-x));
                       return g * sigmoid * (1 - sigmoid);
                   });
}

void Sigmoid::forward() {
    Array *input = prev->get_output();
    sigmoid_forward(input, input);
}

void Sigmoid::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    sigmoid_backward(output_grad, output_grad, input);
}

// Tanh
//
// Tanh derivative: dtanh/dx = 1 - tanh(x)^2
//
// Note that this is a drop-in layer that directly alters previous layer's
// output and next layer's input gradient
void tanh_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "tanh_forward: output size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output->get_vec().begin(), [](float x) { return tanhf(x); });
}

void tanh_backward(Array *input_grad, const Array *output_grad,
                   const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "tanh_backward: input grad size isn't equal to output grad size");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "tanh_backward: input grad size isn't equal to input size");

    std::transform(input->get_vec().begin(), input->get_vec().end(),
                   output_grad->get_vec().begin(),
                   input_grad->get_vec().begin(), [](float x, float g) {
                       float tanh = tanhf(x);
                       return g * (1 - tanh * tanh);
                   });
}

void Tanh::forward() {
    Array *input = prev->get_output();
    tanh_forward(input, input);
}

void Tanh::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    tanh_backward(output_grad, output_grad, input);
}

// Softmax
//
// The activation function of this layer is a folding function, meaning it
// performs aggregation of the input instead of applying the calculation
// separately.
// This is typically used as the last layer of a neural network, normalizing
// output into probability distribution for loss evaluation
void softmax_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "softmax_forward: output size isn't equal to input size");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    for (int i = 0; i < batch_size; i++) {
        auto in_begin = input->get_vec().begin() + i * batch_stride;
        auto out_begin = output->get_vec().begin() + i * batch_stride;

        float max_val = *std::max_element(in_begin, in_begin + batch_stride);
        std::transform(in_begin, in_begin + batch_stride, out_begin,
                       [&](float x) { return expf(x - max_val); });

        float exp_sum =
            std::accumulate(out_begin, out_begin + batch_stride, 0.0);
        for (int k = 0; k < batch_stride; k++) {
            out_begin[k] /= exp_sum;
            out_begin[k] += EPS; // prevent applying log on 0
        }
    }
}

void Softmax::forward() {
    const Array *input = prev->get_output();
    set_array_ptr(output, input->get_shape());
    softmax_forward(output.get(), input);
}

void Softmax::backward() {
    const Array *output_grad = next->get_grad();
    set_array_ptr(grad, output_grad->get_shape());
    std::copy(output_grad->get_vec().begin(), output_grad->get_vec().end(),
              grad->get_vec().begin());
}

//
// LogSoftmax
//
// Similar to softmax, just with log values as output instead of logits
void log_softmax_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_shape(), input->get_shape(),
             "softmax_forward: output shape isn't equal to input shape");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    for (int i = 0; i < batch_size; i++) {
        auto in_begin = input->get_vec().begin() + i * batch_stride;
        auto out_begin = output->get_vec().begin() + i * batch_stride;

        float max_val = *std::max_element(in_begin, in_begin + batch_stride);
        float log_sum = std::accumulate(
            in_begin, in_begin + batch_stride, 0.0,
            [&](float s, float x) { return s + expf(x - max_val); });
        log_sum = max_val + logf(log_sum);

        std::transform(in_begin, in_begin + batch_stride, out_begin,
                       [&](float x) { return x - log_sum; });
    }
}

void log_softmax_backward(Array *input_grad, const Array *output_grad,
                          const Array *input) {
    CHECK_EQ(output_grad->get_shape(), input_grad->get_shape(),
             "softmax_backward: output grad shape isn't equal to input grad "
             "shape");
    CHECK_EQ(output_grad->get_shape(), input->get_shape(),
             "softmax_backward: output grad shape isn't equal to input shape");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    for (int i = 0; i < batch_size; i++) {
        auto in_grad_begin = input_grad->get_vec().begin() + i * batch_stride;
        auto out_grad_begin = output_grad->get_vec().begin() + i * batch_stride;
        auto in_begin = input->get_vec().begin() + i * batch_stride;

        float max_val = *std::max_element(in_begin, in_begin + batch_stride);
        float log_sum = std::accumulate(
            in_begin, in_begin + batch_stride, 0.0,
            [&](float s, float x) { return s + expf(x - max_val); });
        log_sum = max_val + logf(log_sum);

        float dldy_sum =
            std::accumulate(out_grad_begin, out_grad_begin + batch_stride, 0.0);

        std::transform(
            in_begin, in_begin + batch_stride, out_grad_begin, in_grad_begin,
            [&](float x, float g) { return g - dldy_sum * expf(x - log_sum); });
    }
}

void LogSoftmax::forward() {
    const Array *input = prev->get_output();
    set_array_ptr(output, input->get_shape());
    log_softmax_forward(output.get(), input);
}

void LogSoftmax::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();
    set_array_ptr(grad, input->get_shape());
    log_softmax_backward(grad.get(), output_grad, input);
}

} // namespace nnv2