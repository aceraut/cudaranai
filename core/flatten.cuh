#pragma once

#include "layer.cuh"

#include <vector>

namespace nnv2 {

// Flatten layer is a drop-in layer that directly transforms previous layer's
// output to flattened representation (except the first dimension, which denotes
// batch size) in the forward phase, and transforms next layer's input gradient
// to the original shape in the backward phase.
// Since the underlying data is arranged in row-major order, only the shape
// is affected.

class Flatten : public Layer {
public:
  Flatten() : Layer() {}

  void forward() override;
  void backward() override;

  Array *get_output() { return prev->get_output(); }
  const Array *get_output() const { return prev->get_output(); }

  Array *get_grad() { return next->get_grad(); }
  const Array *get_grad() const { return next->get_grad(); }

private:
  ShapeType in_shape;
};

} // namespace nnv2