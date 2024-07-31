// This file implements Loss classes: cross-entropy loss and NLL loss.
//
// Loss layer validates the prediction results with the actual results to
// calculate the loss value and the loss gradient for backpropagation.

#include "common.cuh"
#include "loss.cuh"

namespace nnv2 {

//
// Cross-entropy
//
// Calculates cross-entropy loss value using the formula
// loss = -1/n * sum(sum(y * log(p)))
void cross_entropy_loss(Array *output, const Array *input, const Array *y,
                        ArrayMap &cache) {
    CHECK_EQ(input->get_shape(), y->get_shape(),
             "calculate_loss: shape of input mismatched with y");

    set_array_cache(cache, "log_pred", input->get_shape());
    func_log(cache["log_pred"].get(), input);

    set_array_cache(cache, "loss_sparse", input->get_shape());
    func_mul(cache["loss_sparse"].get(), cache["log_pred"].get(), y);

    // reduce the distribution matrix to one-dimensional array
    set_array_cache(cache, "loss", {input->get_shape()[0]});
    func_sum(cache["loss"].get(), cache["loss_sparse"].get(), 1);

    // calculate average log loss value of a batch
    func_mean(output, cache["loss"].get(), 0, false);
    output->get_vec()[0] *= -1.0;
}

void cross_entropy_loss_backward(Array *input_grad, const Array *input,
                                 const Array *y) {
    CHECK_EQ(input->get_shape(), input_grad->get_shape(),
             "loss_backward: shape of input grad mismatched with input");
    CHECK_EQ(input->get_shape(), y->get_shape(),
             "loss_backward: shape of input mismatched with y");
    func_sub(input_grad, input, y);
}

float CrossEntropyLoss::calculate_loss(const Array *labels) {
    y = labels;

    const Array *input = prev->get_output();
    set_array_ptr(output, {1});
    cross_entropy_loss(output.get(), input, y, cache);

    return output->get_vec()[0];
}

void CrossEntropyLoss::backward() {
    const Array *input = prev->get_output();
    set_array_ptr(grad, input->get_shape());
    cross_entropy_loss_backward(grad.get(), input, y);
}

//
// NLLLoss
//
void nll_loss(Array *output, const Array *input, const Array *y,
              ArrayMap &cache) {
    CHECK_EQ(input->get_shape(), y->get_shape(),
             "calculate_loss: shape of input mismatched with y");

    set_array_cache(cache, "loss_sparse", input->get_shape());
    func_mul(cache["loss_sparse"].get(), input, y);

    // reduce the distribution matrix to one-dimensional array
    set_array_cache(cache, "loss", {input->get_shape()[0]});
    func_sum(cache["loss"].get(), cache["loss_sparse"].get(), 1);

    // calculate average log loss value of a batch
    func_mean(output, cache["loss"].get(), 0, false);
    output->get_vec()[0] *= -1.0;
}

void nll_loss_backward(Array *input_grad, const Array *y) {
    CHECK_EQ(input_grad->get_shape(), y->get_shape(),
             "calculate_loss: shape of input grad mismatched with y");

    int batch_size = y->get_shape()[0];
    func_mul(input_grad, y, -1.0 / batch_size);
}

float NLLLoss::calculate_loss(const Array *labels) {
    y = labels;

    const Array *input = prev->get_output();
    set_array_ptr(output, {1});
    nll_loss(output.get(), input, y, cache);

    return output->get_vec()[0];
}

void NLLLoss::backward() {
    const Array *input = prev->get_output();
    set_array_ptr(grad, input->get_shape());
    nll_loss_backward(grad.get(), y);
}

} // namespace nnv2