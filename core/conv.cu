// This file implements the Convolution layer using the im2col technique
//
// The im2col technique involves transforming the input image into a
// representation where the convolution operation can be efficiently performed
// using matrix multiplication with the kernel.
//
// Specifically speaking, given:
// - The shape of the input image is (i_h, i_w)
// - The shape of the output image is (o_h, o_w)
// - The shape of the kernel is (k_h, k_w)
//
// In the im2col technique, the input image is transformed into an unrolled
// representation with shape (k_h * k_w, o_h * o_w). Each column in this
// transformed array is the flattened representation of a section of the input
// image that intersects with the kernel.

#include "conv.h"
#include "common.h"

#include <utility>
#include <vector>

namespace nnv2 {

// This implementation expands the algorithm described above with support for
// multi-channel batch calculation

// Gets the element at a specific image (with padding)
static float im_get_pixel(const Array *im, int feat_idx, int r, int c,
                          int pad_h, int pad_w) {
    int h = im->get_shape()[2];
    int w = im->get_shape()[3];

    r -= pad_h;
    c -= pad_w;

    if (r < 0 || r >= h || c < 0 || c >= w) {
        return 0;
    }

    int im_idx = (feat_idx * h + r) * w + c;
    return im->get_vec()[im_idx];
}

void im2col(const Array *im, Array *cols, int pad_h, int pad_w, int kernel_h,
            int kernel_w, int stride_h, int stride_w) {
    int batch_size = im->get_shape()[0];
    int im_feats = im->get_shape()[1];
    int im_h = im->get_shape()[2];
    int im_w = im->get_shape()[3];

    int out_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;

    int cols_size = batch_size * im_feats * kernel_h * kernel_w;
    for (int i = 0; i < cols_size; i++) {
        int kernel_r = (i / kernel_w) % kernel_h;
        int kernel_c = i % kernel_w;
        int feat_idx = i / kernel_w / kernel_h;

        for (int r = 0; r < out_h; r++) {
            for (int c = 0; c < out_w; c++) {
                int im_r = kernel_r + stride_h * r;
                int im_c = kernel_c + stride_w * c;
                int idx = (i * out_h + r) * out_w + c;
                cols->get_vec()[idx] =
                    im_get_pixel(im, feat_idx, im_r, im_c, pad_h, pad_w);
            }
        }
    }
}

// The col2im function transforms the unrolled representation into the
// original ``image`` format.
// Note that since elements that have the same index in the original format are
// rolled back as sum and not assignment, this function is only applied to
// the unrolled representation of the input gradient.

// Add value to the element at a specific image (with padding)
static void im_add_pixel(Array *im, int feat_idx, int r, int c, int pad_h,
                         int pad_w, float value) {
    int h = im->get_shape()[2];
    int w = im->get_shape()[3];

    r -= pad_h;
    c -= pad_w;

    if (r < 0 || r >= h || c < 0 || c >= w) {
        return;
    }

    int im_idx = (feat_idx * h + r) * w + c;
    im->get_vec()[im_idx] += value;
}

void col2im(const Array *cols, Array *im, int pad_h, int pad_w, int kernel_h,
            int kernel_w, int stride_h, int stride_w) {
    int batch_size = im->get_shape()[0];
    int im_feats = im->get_shape()[1];
    int im_h = im->get_shape()[2];
    int im_w = im->get_shape()[3];

    int out_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;

    int cols_size = batch_size * im_feats * kernel_h * kernel_w;
    for (int i = 0; i < cols_size; i++) {
        int kernel_r = (i / kernel_w) % kernel_h;
        int kernel_c = i % kernel_w;
        int feat_idx = i / kernel_w / kernel_h;

        for (int r = 0; r < out_h; r++) {
            for (int c = 0; c < out_w; c++) {
                int im_r = kernel_r + stride_h * r;
                int im_c = kernel_c + stride_w * c;
                int idx = (i * out_h + r) * out_w + c;
                float value = cols->get_vec()[idx];
                im_add_pixel(im, feat_idx, im_r, im_c, pad_h, pad_w, value);
            }
        }
    }
}

// This function performs convolution on input and kernel
void conv_forward(Array *output, const Array *input, Array *cols, Array *kernel,
                  int pad_h, int pad_w, int stride_h, int stride_w) {
    CHECK_EQ(output->get_shape().size(), 4, "conv_forward: output shape error");
    CHECK_EQ(input->get_shape().size(), 4, "conv_forward: input shape error");
    CHECK_EQ(cols->get_shape().size(), 3, "conv_forward: cols shape error");
    CHECK_EQ(kernel->get_shape().size(), 4, "conv_forward: kernel shape error");

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];

    CHECK_EQ(output->get_shape()[0], batch_size,
             "conv_forward: batch size error");

    int out_feats = output->get_shape()[1];
    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];

    CHECK_EQ(kernel->get_shape()[0], out_feats,
             "conv_forward: feature size error");
    CHECK_EQ(kernel->get_shape()[1], in_feats,
             "conv_forward: feature size error");

    int kernel_h = kernel->get_shape()[2];
    int kernel_w = kernel->get_shape()[3];

    // Cols = im2col(X)
    // X: shape (n, i_f, i_h, i_w)
    im2col(input, cols, pad_h, pad_w, kernel_h, kernel_w, stride_w, stride_h);

    // Y = K * Cols
    // At this point, the shapes of the involved arrays are
    // Y:    shape (n,   o_f, o_h, o_w)
    // K:    shape (o_f, i_f, k_h, k_w)
    // Cols: shape (n, i_f * k_h * k_w, o_h * o_w)

    // reshape K to (o_f, i_f * k_h * k_w)
    kernel->reshape({out_feats, in_feats * kernel_h * kernel_w});
    // reshape Y to (n, o_f, o_h * o_w)
    output->reshape({batch_size, out_feats, out_h * out_w});
    // calculate Y = K * Cols
    func_matmul(output, kernel, cols, 1);

    // recover shape
    kernel->reshape({out_feats, in_feats, kernel_h, kernel_w});
    output->reshape({batch_size, out_feats, out_h, out_w});
}

// Adds bias based on the output feature index
void conv_forward_bias(Array *output, const Array *bias) {
    CHECK_EQ(bias->get_shape()[0], 1,
             "conv_forward_bias: bias isn't a column vector");

    int out_feats = output->get_shape()[1];
    CHECK_EQ(bias->get_shape()[1], out_feats,
             "conv_forward_bias: bias size "
             "isn't equal to number of output features");

    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];

    std::vector<float> &output_ref = output->get_vec();
    for (int i = 0; i < output_ref.size(); i++) {
        int feat_idx = (i / (out_h * out_w)) % out_feats;
        output_ref[i] += bias->get_vec()[feat_idx];
    }
}

// Calculates the input gradient, kernel gradient from the output gradient
void conv_backward(Array *input_grad, Array *kernel_grad, Array *output_grad,
                   const Array *input, Array *kernel, const Array *cols,
                   int pad_h, int pad_w, int stride_h, int stride_w,
                   ArrayMap &cache) {
    CHECK_EQ(input_grad->get_shape().size(), 4,
             "conv_backward: input gradient shape error");
    CHECK_EQ(kernel_grad->get_shape().size(), 4,
             "conv_backward: input shape error");
    CHECK_EQ(cols->get_shape().size(), 3, "conv_backward: cols shape error");
    CHECK_EQ(output_grad->get_shape().size(), 4,
             "conv_backward: output gradient shape error");

    CHECK_EQ(input->get_shape(), input_grad->get_shape(),
             "conv_backward: shape mismatch between input and its gradient");
    CHECK_EQ(kernel->get_shape(), kernel_grad->get_shape(),
             "conv_backward: shape mismatch between kernel and its gradient");

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];

    CHECK_EQ(output_grad->get_shape()[0], batch_size,
             "conv_backward: batch size error");

    int out_feats = output_grad->get_shape()[1];
    int out_h = output_grad->get_shape()[2];
    int out_w = output_grad->get_shape()[3];

    CHECK_EQ(kernel->get_shape()[0], out_feats,
             "conv_backward: feature size error");
    CHECK_EQ(kernel->get_shape()[1], in_feats,
             "conv_backward: feature size error");

    int kernel_h = kernel->get_shape()[2];
    int kernel_w = kernel->get_shape()[3];

    // As: Y = K * Cols, Cols = im2col(X)
    // => dL/dK = dL/dY * Cols^T
    //    dL/dCols = K^T * dL/dY
    //    dL/dX = col2im(dL/dCols)

    // Calculate dL/dK = dL/dY * Cols^T
    // At this point, the shapes of the involved arrays are
    // dL/dK:   shape (o_f, i_f, k_h, k_w)
    // dL/dY:   shape (n, o_f, o_h, o_w)
    // Cols^T:  shape (n, o_h * o_w, i_f * k_h * k_w)

    // reshape dL/dY to (n, o_f, o_h * o_w)
    output_grad->reshape({batch_size, out_feats, out_h * out_w});

    // Cols^T
    set_array_cache(
        cache, "cols_t",
        {batch_size, out_h * out_w, in_feats * kernel_h * kernel_w});
    func_transpose(cache["cols_t"].get(), cols);

    // dL/dY * Cols^T
    set_array_cache(cache, "kernel_grad_unfolded",
                    {batch_size, out_feats, in_feats * kernel_h * kernel_w});
    func_matmul(cache["kernel_grad_unfolded"].get(), output_grad,
                cache["cols_t"].get());

    // dL/dK is the sum of dL/dY * Cols^T along the batch
    // Since currently dL/dY * Cols^T shape is (n, o_f, i_f * k_h * k_w),
    // reshape dL/dY to (n, o_f, i_f, k_h, k_w)
    cache["kernel_grad_unfolded"]->reshape(
        {batch_size, out_feats, in_feats, kernel_h, kernel_w});
    func_sum(kernel_grad, cache["kernel_grad_unfolded"].get(), 0);

    // Calculate dL/dX from dL/dCols = K^T * dL/dY
    // At this point, the shapes of the involved arrays are
    // dL/dCols: shape (n, i_f * k_h * k_w, o_h * o_w)
    // K:        shape (o_f, i_f, k_h, k_w)
    // dL/dY:    shape (n, o_f, o_h, o_w) => (n, o_f, o_h * o_w)

    // K^T
    // reshape K to (o_f, i_f * k_h * k_w)
    kernel->reshape({out_feats, in_feats * kernel_h * kernel_w});
    set_array_cache(cache, "kernel_t",
                    {in_feats * kernel_h * kernel_w, out_feats});
    func_transpose(cache["kernel_t"].get(), kernel);

    // dL/dCols
    set_array_cache(
        cache, "cols_grad",
        {batch_size, in_feats * kernel_h * kernel_w, out_h * out_w});
    func_matmul(cache["cols_grad"].get(), cache["kernel_t"].get(), output_grad,
                1);

    // dL/dX
    col2im(cache["cols_grad"].get(), input_grad, pad_h, pad_w, kernel_h,
           kernel_w, stride_h, stride_w);

    // restore shape
    output_grad->reshape({batch_size, out_feats, out_h, out_w});
    kernel->reshape({out_feats, in_feats, kernel_h, kernel_w});
}

void conv_backward_bias(Array *bias_grad, const Array *output_grad,
                        ArrayMap &cache) {
    CHECK_EQ(bias_grad->get_shape()[0], 1,
             "conv_backward_bias: bias grad isn't a column vector");

    int batch_size = output_grad->get_shape()[0];
    int out_feats = output_grad->get_shape()[1];
    int out_h = output_grad->get_shape()[2];

    CHECK_EQ(bias_grad->get_shape()[1], out_feats,
             "conv_backward_bias: bias "
             "grad size isn't equal to number of output features");

    set_array_cache(cache, "fold3", {batch_size, out_feats, out_h});
    set_array_cache(cache, "fold2", {batch_size, out_feats});
    func_sum(cache["fold3"].get(), output_grad, 3);
    func_sum(cache["fold2"].get(), cache["fold3"].get(), 2);
    func_sum(bias_grad, cache["fold2"].get(), 0, false);
}

Conv2D::Conv2D(int in_feats, int out_feats, int in_h, int in_w, int pad_h,
               int pad_w, int kernel_h, int kernel_w, int stride_h,
               int stride_w, const Initializer *init)
    : in_feats(in_feats), out_feats(out_feats), in_h(in_h), in_w(in_w),
      pad_h(pad_h), pad_w(pad_w), kernel_h(kernel_h), kernel_w(kernel_w),
      stride_h(stride_h), stride_w(stride_w) {
    kernel.reset(new Array({out_feats, in_feats, kernel_h, kernel_w}));
    kernel_grad.reset(new Array({out_feats, in_feats, kernel_h, kernel_w}));
    bias.reset(new Array({1, out_feats}));
    bias_grad.reset(new Array({1, out_feats}));

    // initialize parameters
    int fan_in = kernel_h * kernel_w * in_feats;
    int fan_out = kernel_h * kernel_w * out_feats;
    init->initialize(kernel.get(), fan_in, fan_out);
    init->initialize(bias.get(), fan_in, fan_out);
}

std::vector<Param> Conv2D::get_parameters() {
    return {std::make_pair(kernel.get(), kernel_grad.get()),
            std::make_pair(bias.get(), bias_grad.get())};
}

void Conv2D::forward() {
    const Array *input = prev->get_output();

    int batch_size = input->get_shape()[0];
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    set_array_ptr(output, {batch_size, out_feats, out_h, out_w});
    set_array_ptr(imcols,
                  {batch_size, in_feats * kernel_h * kernel_w, out_h * out_w});

    conv_forward(output.get(), input, imcols.get(), kernel.get(), pad_h, pad_w,
                 stride_h, stride_w);
    conv_forward_bias(output.get(), bias.get());
}

void Conv2D::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();

    set_array_ptr(grad, input->get_shape());

    conv_backward_bias(bias_grad.get(), output_grad, cache);
    conv_backward(grad.get(), kernel_grad.get(), output_grad, input,
                  kernel.get(), imcols.get(), pad_h, pad_w, stride_h, stride_w,
                  cache);
}

} // namespace nnv2