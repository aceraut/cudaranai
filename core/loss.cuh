#pragma once

#include "common.cuh"
#include "layer.cuh"

namespace nnv2 {

class Loss : public Layer {
public:
  Loss() : Layer() {}
  virtual ~Loss() {}

  virtual float calculate_loss(const Array *y) = 0;

protected:
  ArrayMap cache;
  const Array *y;
};

// Only pair this with Softmax layer
class CrossEntropyLoss : public Loss {
public:
  CrossEntropyLoss() : Loss() {}
  float calculate_loss(const Array *y) override;
  void backward() override;
};

// Only pair this with LogSoftmax layer
class NLLLoss : public Loss {
public:
  NLLLoss() : Loss() {}
  float calculate_loss(const Array *y) override;
  void backward() override;
};

// Helper functions
void cross_entropy_loss(
    Array *output,
    const Array *input,
    const Array *y,
    ArrayMap &cache);
void cross_entropy_loss_backward(
    Array *input_grad,
    const Array *input,
    const Array *y);

void nll_loss(
    Array *output,
    const Array *input,
    const Array *y,
    ArrayMap &cache);
void nll_loss_backward(Array *input_grad, const Array *y);

} // namespace nnv2