#include "common.cuh"
#include "maxpool.cuh"

#include <cfloat>
#include <cmath>

#include <cuda_runtime.h>

namespace nnv2 {

// The forward phase of max-pooling layer involves selecting the pixel with the
// largest value in every smaller local patches of the input feature maps and
// place them in the output feature maps.
__global__ void maxpool_forward_kernel(
    float *output,
    int *max_indices,
    const float *input,
    int in_h,
    int in_w,
    int pad_h,
    int pad_w,
    int filter_h,
    int filter_w,
    int stride_h,
    int stride_w,
    int out_h,
    int out_w) {
  // Each thread handles a pixel in the output image
  // Point to input and output images that this thread handles
  int feat_idx = blockIdx.x;
  input += feat_idx * in_h * in_w;
  output += feat_idx * out_h * out_w;
  max_indices += feat_idx * out_h * out_w;

  int out_x_start = blockIdx.y * blockDim.y + threadIdx.y;
  int out_y_start = threadIdx.x;

  int out_x_stride = gridDim.y * blockDim.y;
  int out_y_stride = blockDim.x;

  for (int out_x = out_x_start; out_x < out_h; out_x += out_x_stride) {
    for (int out_y = out_y_start; out_y < out_w; out_y += out_y_stride) {
      // Locate the base coords of the section in the input image that
      // affects (out_x, out_y) of the output image
      int in_x_start = out_x * stride_h - pad_h;
      int in_y_start = out_y * stride_w - pad_w;

      int in_x_end = in_x_start + filter_h;
      in_x_end = in_x_end > in_h ? in_h : in_x_end;
      int in_y_end = in_y_start + filter_w;
      in_y_end = in_y_end > in_w ? in_w : in_y_end;

      in_x_start = in_x_start < 0 ? 0 : in_x_start;
      in_y_start = in_y_start < 0 ? 0 : in_y_start;

      // Loop over the local patch and select the pixel with largest value
      int idx = out_x * out_w + out_y;
      float max_val = -FLT_MAX;
      float max_idx = -1;

      for (int in_x = in_x_start; in_x < in_x_end; in_x++) {
        for (int in_y = in_y_start; in_y < in_y_end; in_y++) {
          int in_idx = in_x * in_w + in_y;
          if (input[in_idx] > max_val) {
            max_idx = in_idx;
            max_val = input[in_idx];
          }
        }
      }
      output[idx] = max_val;
      max_indices[idx] = max_idx;
    }
  }
}

void maxpool_forward(
    Array *output,
    const Array *input,
    VecType<int> &indices,
    int pad_h,
    int pad_w,
    int filter_h,
    int filter_w,
    int stride_h,
    int stride_w) {
  const ShapeType &output_shape = output->get_shape();
  const ShapeType &input_shape = input->get_shape();

  CHECK_EQ(output_shape.size(), 4, "maxpool_forward: output shape error");
  CHECK_EQ(input_shape.size(), 4, "maxpool_forward: input shape error");
  CHECK_EQ(
      indices.size(),
      output->get_vec().size(),
      "maxpool_forward: size mismatch beetween indices and output");

  int batch_size = input_shape[0];
  int in_feats = input_shape[1];

  CHECK_EQ(output_shape[0], batch_size, "maxpool_forward: batch size error");
  CHECK_EQ(output_shape[1], in_feats, "maxpool_forward: feature size error");

  int in_h = input_shape[2];
  int in_w = input_shape[3];
  int out_h = output_shape[2];
  int out_w = output_shape[3];

  float *output_raw = RAW_PTR(output->get_vec());
  int *indices_raw = RAW_PTR(indices);
  const float *input_raw = RAW_PTR(input->get_vec());

  int block_num = std::max(1, 16 / in_feats);
  dim3 grid_dim(batch_size * in_feats, block_num);
  dim3 block_dim(32, 8);

  maxpool_forward_kernel<<<grid_dim, block_dim>>>(
      output_raw,
      indices_raw,
      input_raw,
      in_h,
      in_w,
      pad_h,
      pad_w,
      filter_h,
      filter_w,
      stride_h,
      stride_w,
      out_h,
      out_w);
  CUDA_POST_KERNEL_CHECK;
}

// Each output gradient is propagated back to its pooled position in the input.
__global__ void maxpool_backward_kernel(
    float *input_grad,
    const float *output_grad,
    const int *max_indices,
    int in_h,
    int in_w,
    int out_h,
    int out_w) {
  int feat_idx = blockIdx.x;
  input_grad += feat_idx * in_h * in_w;
  output_grad += feat_idx * out_h * out_w;
  max_indices += feat_idx * out_h * out_w;

  int out_x_start = blockIdx.y * blockDim.y + threadIdx.y;
  int out_y_start = threadIdx.x;

  int out_x_stride = gridDim.y * blockDim.y;
  int out_y_stride = blockDim.x;

  for (int out_x = out_x_start; out_x < out_h; out_x += out_x_stride) {
    for (int out_y = out_y_start; out_y < out_w; out_y += out_y_stride) {
      int idx = out_x * out_w + out_y;
      int max_idx = max_indices[idx];

      input_grad[max_idx] += output_grad[idx];
    }
  }
}

__global__ void maxpool_backward_atomic_kernel(
    float *input_grad,
    const float *output_grad,
    const int *max_indices,
    int in_h,
    int in_w,
    int out_h,
    int out_w) {
  int feat_idx = blockIdx.x;
  input_grad += feat_idx * in_h * in_w;
  output_grad += feat_idx * out_h * out_w;
  max_indices += feat_idx * out_h * out_w;

  int out_x_start = blockIdx.y * blockDim.y + threadIdx.y;
  int out_y_start = threadIdx.x;

  int out_x_stride = gridDim.y * blockDim.y;
  int out_y_stride = blockDim.x;

  for (int out_x = out_x_start; out_x < out_h; out_x += out_x_stride) {
    for (int out_y = out_y_start; out_y < out_w; out_y += out_y_stride) {
      int idx = out_x * out_w + out_y;
      int max_idx = max_indices[idx];

      atomicAdd(&input_grad[max_idx], output_grad[idx]);
    }
  }
}

void maxpool_backward(
    Array *input_grad,
    const Array *output_grad,
    const VecType<int> &indices,
    int pad_h,
    int pad_w,
    int filter_h,
    int filter_w,
    int stride_h,
    int stride_w) {
  const ShapeType &input_grad_shape = input_grad->get_shape();
  const ShapeType &output_grad_shape = output_grad->get_shape();

  CHECK_EQ(
      input_grad_shape.size(),
      4,
      "maxpool_backward: input gradient shape error");
  CHECK_EQ(
      output_grad_shape.size(),
      4,
      "maxpool_backward: output gradient shape error");
  CHECK_EQ(
      indices.size(),
      output_grad->get_vec().size(),
      "maxpool_backward: size mismatch between indices and output grad");

  int batch_size = input_grad_shape[0];
  int in_feats = input_grad_shape[1];

  CHECK_EQ(
      output_grad_shape[0], batch_size, "maxpool_backward: batch size error");
  CHECK_EQ(
      output_grad_shape[1], in_feats, "maxpool_backward: feature size error");

  float *input_grad_raw = RAW_PTR(input_grad->get_vec());
  const float *output_grad_raw = RAW_PTR(output_grad->get_vec());
  const int *indices_raw = RAW_PTR(indices);

  int in_h = input_grad_shape[2];
  int in_w = input_grad_shape[3];
  int out_h = output_grad_shape[2];
  int out_w = output_grad_shape[3];

  int block_num = std::max(1, 16 / in_feats);
  dim3 grid_dim(batch_size * in_feats, block_num);
  dim3 block_dim(32, 8);

  // If stride dimension is the same as the filter dimension, gradient value
  // of pooled pixels in the input can be propagated without atomic expression
  if (filter_h == stride_h && filter_w == stride_w) {
    maxpool_backward_kernel<<<grid_dim, block_dim>>>(
        input_grad_raw, output_grad_raw, indices_raw, in_h, in_w, out_h, out_w);
  } else {
    maxpool_backward_atomic_kernel<<<grid_dim, block_dim>>>(
        input_grad_raw, output_grad_raw, indices_raw, in_h, in_w, out_h, out_w);
  }
  CUDA_POST_KERNEL_CHECK;
}

MaxPool2D::MaxPool2D(
    int pad_h,
    int pad_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w)
    : pad_h(pad_h), pad_w(pad_w), kernel_h(kernel_h), kernel_w(kernel_w),
      stride_h(stride_h), stride_w(stride_w) {}

void MaxPool2D::forward() {
  const Array *input = prev->get_output();
  const ShapeType &input_shape = input->get_shape();

  int batch_size = input_shape[0];
  int in_feats = input_shape[1];
  int in_h = input_shape[2];
  int in_w = input_shape[3];
  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  utils::set_array_ptr(output, {batch_size, in_feats, out_h, out_w});
  indices.resize(output->get_vec().size());

  maxpool_forward(
      output.get(),
      input,
      indices,
      pad_h,
      pad_w,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w);
}

void MaxPool2D::backward() {
  const Array *input = prev->get_output();
  const Array *output_grad = next->get_grad();

  utils::set_array_ptr(grad, input->get_shape());

  maxpool_backward(
      grad.get(),
      output_grad,
      indices,
      pad_h,
      pad_w,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w);
}

} // namespace nnv2