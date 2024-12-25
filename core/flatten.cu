#include "flatten.cuh"

namespace nnv2 {

void Flatten::forward() {
  Array *input = prev->get_output();
  in_shape = input->get_shape();

  int out_feats = 1;
  for (int i = 1; i < in_shape.size(); i++) {
    out_feats *= in_shape[i];
  }

  input->reshape({in_shape[0], out_feats});
}

void Flatten::backward() {
  Array *output_grad = next->get_grad();
  output_grad->reshape(in_shape);
}

} // namespace nnv2