#pragma once

#include "common.cuh"
#include "initializer.cuh"
#include "layer.cuh"

#include <memory>
#include <vector>

namespace nnv2 {

class Conv2D : public Layer {
public:
  Conv2D(int in_feats, int out_feats, int in_h, int in_w, int pad_h, int pad_w,
         int filter_h, int filter_w, int stride_h, int stride_w,
         const Initializer *init);

  std::vector<Param> get_parameters() override;

  void forward() override;
  void backward() override;

private:
  int in_feats;
  int out_feats;

  int in_h;
  int in_w;
  int pad_h;
  int pad_w;
  int filter_h;
  int filter_w;
  int stride_h;
  int stride_w;

  std::unique_ptr<Array> filter;
  std::unique_ptr<Array> filter_grad;

  std::unique_ptr<Array> bias;
  std::unique_ptr<Array> bias_grad;

  std::unique_ptr<Array> col;

  ArrayMap cache;
};

void im2col(const Array *im, Array *col, int pad_h, int pad_w, int filter_h,
            int filter_w, int stride_h, int stride_w);

void col2im(const Array *col, Array *im, int pad_h, int pad_w, int filter_h,
            int filter_w, int stride_h, int stride_w);

void conv_forward(Array *output, const Array *input, Array *col, Array *filter,
                  int pad_h, int pad_w, int stride_h, int stride_w);

void conv_forward_bias(Array *output, const Array *bias);

void conv_backward(Array *input_grad, Array *filter_grad, Array *output_grad,
                   const Array *input, Array *filter, const Array *col,
                   int pad_h, int pad_w, int stride_h, int stride_w,
                   ArrayMap &cache);

void conv_backward_bias(Array *bias_grad, const Array *output_grad,
                        ArrayMap &cache);

} // namespace nnv2