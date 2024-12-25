#include "common.cuh"
#include "loss.cuh"

// Loss layer validates the prediction results with the actual results to
// calculate the loss value and the loss gradient for backpropagation.

namespace nnv2 {

// Cross-entropy is a function to calculate loss value using the formula:
// Loss(P) = - mean(sum(Y * log(P))),
// where P is the output of the forward phase of the classifier with Softmax as
// the final layer and Y is the actual result in one-hot encoding.
void cross_entropy_loss(Array *output, const Array *input, const Array *y,
                        ArrayMap &cache) {
  const ShapeType &input_shape = input->get_shape();

  CHECK_EQ(input_shape, y->get_shape(),
           "cross_entropy_loss: shape mismatch between input and y");

  utils::set_array_cache(cache, "log_pred", input_shape);
  ops::log(cache["log_pred"].get(), input);

  utils::set_array_cache(cache, "loss_sparse", input_shape);
  ops::multiply(cache["loss_sparse"].get(), cache["log_pred"].get(), y);

  // Reduce the distribution matrix on each batch element
  utils::set_array_cache(cache, "loss", {input_shape[0]});
  ops::sum(cache["loss"].get(), cache["loss_sparse"].get(), 1);

  // Calculate average log loss value of a batch
  ops::mean(output, cache["loss"].get(), 0, false);
  output->get_vec()[0] *= -1.0;
}

void cross_entropy_loss_backward(Array *input_grad, const Array *input,
                                 const Array *y) {
  CHECK_EQ(input->get_shape(), input_grad->get_shape(),
           "cross_entropy_loss_backward: shape mismatch between input and "
           "its grad");
  CHECK_EQ(input->get_shape(), y->get_shape(),
           "cross_entropy_loss_backward: shape mismatch between input and y");

  ops::subtract(input_grad, input, y);
}

float CrossEntropyLoss::calculate_loss(const Array *labels) {
  y = labels;

  const Array *input = prev->get_output();
  utils::set_array_ptr(output, {1});

  cross_entropy_loss(output.get(), input, y, cache);
  return output->get_vec()[0];
}

void CrossEntropyLoss::backward() {
  const Array *input = prev->get_output();
  utils::set_array_ptr(grad, input->get_shape());

  cross_entropy_loss_backward(grad.get(), input, y);
}

// Negative log likelihood loss (NLLLoss) is a function to calculate loss value
// using the formula:
// Loss(P) = - mean(sum(Y * P)),
// where P is the output of the forward phase of the classifier with LogSoftmax
// as the final layer and Y is the actual labels in one-hot encoding.
void nll_loss(Array *output, const Array *input, const Array *y,
              ArrayMap &cache) {
  const ShapeType &input_shape = input->get_shape();

  CHECK_EQ(input_shape, y->get_shape(),
           "nll_loss: shape mismatch between input and y");

  utils::set_array_cache(cache, "loss_sparse", input_shape);
  ops::multiply(cache["loss_sparse"].get(), input, y);

  // Reduce the distribution matrix on each batch element
  utils::set_array_cache(cache, "loss", {input_shape[0]});
  ops::sum(cache["loss"].get(), cache["loss_sparse"].get(), 1);

  // Calculate average log loss value of a batch
  ops::mean(output, cache["loss"].get(), 0, false);
  output->get_vec()[0] *= -1.0;
}

void nll_loss_backward(Array *input_grad, const Array *y) {
  CHECK_EQ(input_grad->get_shape(), y->get_shape(),
           "nll_loss_backward: shape mismatch between input and its grad");

  int batch_size = y->get_shape()[0];
  ops::multiply(input_grad, y, -1.0 / batch_size);
}

float NLLLoss::calculate_loss(const Array *labels) {
  y = labels;

  const Array *input = prev->get_output();
  utils::set_array_ptr(output, {1});

  nll_loss(output.get(), input, y, cache);
  return output->get_vec()[0];
}

void NLLLoss::backward() {
  const Array *input = prev->get_output();
  utils::set_array_ptr(grad, input->get_shape());

  nll_loss_backward(grad.get(), y);
}

} // namespace nnv2