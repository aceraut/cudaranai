// This file implements the Convolution layer using the im2col technique
//
// The im2col technique involves transforming the input image into a
// representation where the convolution operation can be efficiently performed
// using matrix multiplication with the filter.
//
// Specifically speaking, given:
// - The shape of the input image is (i_h, i_w)
// - The shape of the output image is (o_h, o_w)
// - The shape of the filter is (k_h, k_w)
//
// In the im2col technique, the input image is transformed into an unrolled
// representation with shape (k_h * k_w, o_h * o_w). Each column in this
// transformed array is the flattened representation of a section of the input
// image that intersects with the filter.

#include "common.cuh"
#include "conv.cuh"

#include <algorithm>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace nnv2 {

__global__ void im2col_kernel(int size, const float *im, float *col, int in_h,
                              int in_w, int pad_h, int pad_w, int filter_h,
                              int filter_w, int stride_h, int stride_w,
                              int out_h, int out_w, int im_stride,
                              int col_stride) {
    // Each thread handles a flattened column of size filter_h * filter_w
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // base coords of top-left element of image on a filter
        int out_x = (idx / out_w) % out_h;
        int out_y = idx % out_w;
        int in_x_start = out_x * stride_h - pad_h;
        int in_y_start = out_y * stride_w - pad_w;

        // point to image and filter column that this thread handles
        int feat_idx = blockIdx.y;
        im += feat_idx * im_stride;
        col += feat_idx * col_stride;

        for (int i = 0; i < filter_h; i++) {
            for (int j = 0; j < filter_w; j++) {
                int in_x = in_x_start + i;
                int in_y = in_y_start + j;

                if (in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    col[idx] = im[in_x * in_h + in_y];
                } else {
                    col[idx] = 0;
                }
                // unrolled image width is out_h * out_w = size
                col += size;
            }
        }
    }
}

void im2col(const Array *im, Array *col, int pad_h, int pad_w, int filter_h,
            int filter_w, int stride_h, int stride_w) {
    int batch_size = im->get_shape()[0];
    int im_feats = im->get_shape()[1];
    int im_h = im->get_shape()[2];
    int im_w = im->get_shape()[3];

    // launch kernels
    const float *im_raw = RAW_PTR(im->get_vec());
    float *col_raw = RAW_PTR(col->get_vec());

    int out_h = (im_h + 2 * pad_h - filter_h) / stride_h + 1;
    int out_w = (im_w + 2 * pad_w - filter_w) / stride_w + 1;
    int im_stride = im_h * im_w;
    int col_stride = filter_h * filter_w * out_h * out_w;

    int size = out_h * out_w;
    dim3 grid_dim(ceil((float)size / BLOCK_SIZE), batch_size * im_feats, 1);

    im2col_kernel<<<grid_dim, BLOCK_SIZE>>>(
        size, im_raw, col_raw, im_h, im_w, pad_h, pad_w, filter_h, filter_w,
        stride_h, stride_w, out_h, out_w, im_stride, col_stride);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void col2im_kernel(int size, float *im, const float *col, int in_h,
                              int in_w, int pad_h, int pad_w, int filter_h,
                              int filter_w, int stride_h, int stride_w,
                              int out_h, int out_w, int im_stride,
                              int col_stride) {
    // each thread handles a pixel in the input image
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // point to image and filter column that this thread handles
        int feat_idx = blockIdx.y;
        im += feat_idx * im_stride;
        col += feat_idx * col_stride;

        // coord in input image
        int in_x = (idx / in_w) % in_h + pad_h;
        int in_y = idx % in_w + pad_w;

        // locate the base coords of the section in the output image that
        // depends on pixel (r, c) of the input image
        int out_x_start =
            (in_x < filter_h) ? 0 : (in_x - filter_h) / stride_h + 1;
        int out_x_end = fminf(out_h, in_x / stride_h + 1);
        int out_y_start =
            (in_y < filter_w) ? 0 : (in_y - filter_w) / stride_w + 1;
        int out_y_end = fminf(out_w, in_y / stride_w + 1);

        float value = 0;
        for (int out_x = out_x_start; out_x < out_x_end; out_x++) {
            for (int out_y = out_y_start; out_y < out_y_end; out_y++) {
                // locate the filter position that contains this pixel
                int filter_x = in_x - stride_h * out_x;
                int filter_y = in_y - stride_w * out_y;

                // find index in column
                int col_idx =
                    ((filter_x * filter_w + filter_y) * out_h + out_x) * out_w +
                    out_y;
                value += col[col_idx];
            }
        }
        im[idx] = value;
    }
}

// The col2im function transforms the unrolled representation into the
// original ``image`` format.
// Note that since elements that have the same index in the original format are
// rolled back as sum and not assignment.
void col2im(const Array *col, Array *im, int pad_h, int pad_w, int filter_h,
            int filter_w, int stride_h, int stride_w) {
    int batch_size = im->get_shape()[0];
    int im_feats = im->get_shape()[1];
    int im_h = im->get_shape()[2];
    int im_w = im->get_shape()[3];

    // launch kernels
    float *im_raw = RAW_PTR(im->get_vec());
    const float *col_raw = RAW_PTR(col->get_vec());

    int out_h = (im_h + 2 * pad_h - filter_h) / stride_h + 1;
    int out_w = (im_w + 2 * pad_w - filter_w) / stride_w + 1;
    int im_stride = im_h * im_w;
    int col_stride = filter_h * filter_w * out_h * out_w;

    int size = im_h * im_w;
    dim3 grid_dim(ceil((float)size / BLOCK_SIZE), batch_size * im_feats, 1);

    col2im_kernel<<<grid_dim, BLOCK_SIZE>>>(
        size, im_raw, col_raw, im_h, im_w, pad_h, pad_w, filter_h, filter_w,
        stride_h, stride_w, out_h, out_w, im_stride, col_stride);
    CUDA_POST_KERNEL_CHECK;
}

// This function performs convolution on input and filter
void conv_forward(Array *output, const Array *input, Array *col, Array *filter,
                  int pad_h, int pad_w, int stride_h, int stride_w) {
    CHECK_EQ(output->get_shape().size(), 4, "conv_forward: output shape error");
    CHECK_EQ(input->get_shape().size(), 4, "conv_forward: input shape error");
    CHECK_EQ(col->get_shape().size(), 3, "conv_forward: col shape error");
    CHECK_EQ(filter->get_shape().size(), 4, "conv_forward: filter shape error");

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];

    CHECK_EQ(output->get_shape()[0], batch_size,
             "conv_forward: batch size error");

    int out_feats = output->get_shape()[1];
    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];

    CHECK_EQ(filter->get_shape()[0], out_feats,
             "conv_forward: feature size error");
    CHECK_EQ(filter->get_shape()[1], in_feats,
             "conv_forward: feature size error");

    int filter_h = filter->get_shape()[2];
    int filter_w = filter->get_shape()[3];

    // Cols = im2col(X)
    // X: shape (n, i_f, i_h, i_w)
    im2col(input, col, pad_h, pad_w, filter_h, filter_w, stride_w, stride_h);

    // Y = K * Cols
    // At this point, the shapes of the involved arrays are
    // Y:    shape (n,   o_f, o_h, o_w)
    // K:    shape (o_f, i_f, k_h, k_w)
    // Cols: shape (n, i_f * k_h * k_w, o_h * o_w)

    // reshape K to (o_f, i_f * k_h * k_w)
    filter->reshape({out_feats, in_feats * filter_h * filter_w});
    // reshape Y to (n, o_f, o_h * o_w)
    output->reshape({batch_size, out_feats, out_h * out_w});
    // calculate Y = K * Cols
    func_matmul(output, filter, col, 1);

    // recover shape
    filter->reshape({out_feats, in_feats, filter_h, filter_w});
    output->reshape({batch_size, out_feats, out_h, out_w});
}

__global__ void conv_forward_bias_kernel(int size, float *output,
                                         const float *bias, int im_stride,
                                         int out_feats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = (idx / im_stride) % out_feats;
        output[idx] += bias[bias_idx];
    }
}

// Adds bias based on the output feature index
void conv_forward_bias(Array *output, const Array *bias) {
    CHECK_EQ(bias->get_shape()[0], 1,
             "conv_forward_bias: bias isn't a column vector");

    int out_feats = output->get_shape()[1];
    CHECK_EQ(bias->get_shape()[1], out_feats,
             "conv_forward_bias: mismatch between bias size and number of "
             "output features");

    int batch_size = output->get_shape()[0];
    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];

    float *output_raw = RAW_PTR(output->get_vec());
    const float *bias_raw = RAW_PTR(bias->get_vec());

    int size = output->get_vec().size();
    int grid_size = ceil((float)size / BLOCK_SIZE);

    conv_forward_bias_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, output_raw, bias_raw, out_h * out_w, out_feats);
    CUDA_POST_KERNEL_CHECK;
}

// Calculates the input gradient, filter gradient from the output gradient
void conv_backward(Array *input_grad, Array *filter_grad, Array *output_grad,
                   const Array *input, Array *filter, const Array *col,
                   int pad_h, int pad_w, int stride_h, int stride_w,
                   ArrayMap &cache) {
    CHECK_EQ(input_grad->get_shape().size(), 4,
             "conv_backward: input gradient shape error");
    CHECK_EQ(filter_grad->get_shape().size(), 4,
             "conv_backward: input shape error");
    CHECK_EQ(col->get_shape().size(), 3, "conv_backward: col shape error");
    CHECK_EQ(output_grad->get_shape().size(), 4,
             "conv_backward: output gradient shape error");

    CHECK_EQ(input->get_shape(), input_grad->get_shape(),
             "conv_backward: shape mismatch between input and its gradient");
    CHECK_EQ(filter->get_shape(), filter_grad->get_shape(),
             "conv_backward: shape mismatch between filter and its gradient");

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];

    CHECK_EQ(output_grad->get_shape()[0], batch_size,
             "conv_backward: batch size error");

    int out_feats = output_grad->get_shape()[1];
    int out_h = output_grad->get_shape()[2];
    int out_w = output_grad->get_shape()[3];

    CHECK_EQ(filter->get_shape()[0], out_feats,
             "conv_backward: feature size error");
    CHECK_EQ(filter->get_shape()[1], in_feats,
             "conv_backward: feature size error");

    int filter_h = filter->get_shape()[2];
    int filter_w = filter->get_shape()[3];

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
        cache, "col_t",
        {batch_size, out_h * out_w, in_feats * filter_h * filter_w});
    func_transpose(cache["col_t"].get(), col);

    // dL/dY * Cols^T
    set_array_cache(cache, "filter_grad_unfolded",
                    {batch_size, out_feats, in_feats * filter_h * filter_w});
    func_matmul(cache["filter_grad_unfolded"].get(), output_grad,
                cache["col_t"].get());

    // dL/dK is the sum of dL/dY * Cols^T along the batch
    // Since currently dL/dY * Cols^T shape is (n, o_f, i_f * k_h * k_w),
    // reshape dL/dY to (n, o_f, i_f, k_h, k_w)
    cache["filter_grad_unfolded"]->reshape(
        {batch_size, out_feats, in_feats, filter_h, filter_w});
    func_sum(filter_grad, cache["filter_grad_unfolded"].get(), 0);

    // Calculate dL/dX from dL/dCols = K^T * dL/dY
    // At this point, the shapes of the involved arrays are
    // dL/dCols: shape (n, i_f * k_h * k_w, o_h * o_w)
    // K:        shape (o_f, i_f, k_h, k_w)
    // dL/dY:    shape (n, o_f, o_h, o_w) => (n, o_f, o_h * o_w)

    // K^T
    // reshape K to (o_f, i_f * k_h * k_w)
    filter->reshape({out_feats, in_feats * filter_h * filter_w});
    set_array_cache(cache, "filter_t",
                    {in_feats * filter_h * filter_w, out_feats});
    func_transpose(cache["filter_t"].get(), filter);

    // dL/dCols
    set_array_cache(
        cache, "col_grad",
        {batch_size, in_feats * filter_h * filter_w, out_h * out_w});
    func_matmul(cache["col_grad"].get(), cache["filter_t"].get(), output_grad,
                1);

    // dL/dX
    col2im(cache["col_grad"].get(), input_grad, pad_h, pad_w, filter_h,
           filter_w, stride_h, stride_w);

    // restore shape
    output_grad->reshape({batch_size, out_feats, out_h, out_w});
    filter->reshape({out_feats, in_feats, filter_h, filter_w});
}

void conv_backward_bias(Array *bias_grad, const Array *output_grad,
                        ArrayMap &cache) {
    CHECK_EQ(bias_grad->get_shape()[0], 1,
             "conv_backward_bias: bias grad isn't a column vector");

    int batch_size = output_grad->get_shape()[0];
    int out_feats = output_grad->get_shape()[1];
    int out_h = output_grad->get_shape()[2];

    CHECK_EQ(
        bias_grad->get_shape()[1], out_feats,
        "conv_backward_bias: mismatch between bias grad size and number of "
        "output features");

    set_array_cache(cache, "fold3", {batch_size, out_feats, out_h});
    set_array_cache(cache, "fold2", {batch_size, out_feats});
    func_sum(cache["fold3"].get(), output_grad, 3);
    func_sum(cache["fold2"].get(), cache["fold3"].get(), 2);
    func_sum(bias_grad, cache["fold2"].get(), 0, false);
}

Conv2D::Conv2D(int in_feats, int out_feats, int in_h, int in_w, int pad_h,
               int pad_w, int filter_h, int filter_w, int stride_h,
               int stride_w, const Initializer *init)
    : in_feats(in_feats), out_feats(out_feats), in_h(in_h), in_w(in_w),
      pad_h(pad_h), pad_w(pad_w), filter_h(filter_h), filter_w(filter_w),
      stride_h(stride_h), stride_w(stride_w) {
    filter.reset(new Array({out_feats, in_feats, filter_h, filter_w}));
    filter_grad.reset(new Array({out_feats, in_feats, filter_h, filter_w}));
    bias.reset(new Array({1, out_feats}));
    bias_grad.reset(new Array({1, out_feats}));

    // initialize parameters
    int fan_in = filter_h * filter_w * in_feats;
    int fan_out = filter_h * filter_w * out_feats;
    init->initialize(filter.get(), fan_in, fan_out);
    init->initialize(bias.get(), fan_in, fan_out);
}

std::vector<Param> Conv2D::get_parameters() {
    return {std::make_pair(filter.get(), filter_grad.get()),
            std::make_pair(bias.get(), bias_grad.get())};
}

void Conv2D::forward() {
    const Array *input = prev->get_output();

    int batch_size = input->get_shape()[0];
    int out_h = (in_h + 2 * pad_h - filter_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - filter_w) / stride_w + 1;

    set_array_ptr(output, {batch_size, out_feats, out_h, out_w});
    set_array_ptr(col,
                  {batch_size, in_feats * filter_h * filter_w, out_h * out_w});

    conv_forward(output.get(), input, col.get(), filter.get(), pad_h, pad_w,
                 stride_h, stride_w);
    conv_forward_bias(output.get(), bias.get());
}

void Conv2D::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();

    set_array_ptr(grad, input->get_shape());

    conv_backward_bias(bias_grad.get(), output_grad, cache);
    conv_backward(grad.get(), filter_grad.get(), output_grad, input,
                  filter.get(), col.get(), pad_h, pad_w, stride_h, stride_w,
                  cache);
}

} // namespace nnv2