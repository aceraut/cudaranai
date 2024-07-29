#pragma once

#include "common.h"
#include "layer.h"

#include <vector>

namespace nnv2 {

class MaxPool2D : public Layer {
public:
    MaxPool2D(int pad_h, int pad_w, int kernel_h, int kernel_w, int stride_h,
              int stride_w);

    void forward() override;
    void backward() override;

private:
    int pad_h;
    int pad_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    std::vector<int> indices;
};

void maxpool_forward(Array *output, const Array *input,
                     std::vector<int> &indices, int pad_h, int pad_w,
                     int kernel_h, int kernel_w, int stride_h, int stride_w);

void maxpool_backward(Array *input_grad, const Array *output_grad,
                      const std::vector<int> &indices);

} // namespace nnv2