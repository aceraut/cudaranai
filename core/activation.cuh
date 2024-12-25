#pragma once

#include "common.cuh"
#include "layer.cuh"

namespace nnv2 {

// ReLU hidden layer activation
class ReLU : public Layer {
public:
  ReLU() : Layer() {}

  void forward() override;
  void backward() override;

  Array *get_output() { return prev->get_output(); }
  const Array *get_output() const { return prev->get_output(); }

  Array *get_grad() { return next->get_grad(); }
  const Array *get_grad() const { return next->get_grad(); }
};

// Sigmoid hidden layer activation
class Sigmoid : public Layer {
public:
  Sigmoid() : Layer() {}

  void forward() override;
  void backward() override;

  Array *get_output() { return prev->get_output(); }
  const Array *get_output() const { return prev->get_output(); }

  Array *get_grad() { return next->get_grad(); }
  const Array *get_grad() const { return next->get_grad(); }
};

// Tanh hidden layer activation
class Tanh : public Layer {
public:
  Tanh() : Layer() {}

  void forward() override;
  void backward() override;

  Array *get_output() { return prev->get_output(); }
  const Array *get_output() const { return prev->get_output(); }

  Array *get_grad() { return next->get_grad(); }
  const Array *get_grad() const { return next->get_grad(); }
};

// Softmax output activation
class Softmax : public Layer {
public:
  Softmax() : Layer() {}

  void forward() override;
  void backward() override;
};

// LogSoftmax output activation
class LogSoftmax : public Layer {
public:
  LogSoftmax() : Layer() {}

  void forward() override;
  void backward() override;
};

// Helper functions
void relu_forward(Array *output, const Array *input);
void relu_backward(Array *input_grad, const Array *output_grad,
                   const Array *input);

void sigmoid_forward(Array *output, const Array *input);
void sigmoid_backward(Array *input_grad, const Array *output_grad,
                      const Array *input);

void tanh_forward(Array *output, const Array *input);
void tanh_backward(Array *input_grad, const Array *output_grad,
                   const Array *input);

void softmax_forward(Array *output, const Array *input);
void softmax_backward(Array *input_grad, const Array *output_grad);

void log_softmax_forward(Array *output, const Array *input);
void log_softmax_backward(Array *input_grad, const Array *output_grad,
                          const Array *input);

} // namespace nnv2