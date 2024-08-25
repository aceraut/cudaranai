#include "activation.cuh"
#include "common.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

namespace nnv2 {

void relu_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "relu_forward: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [] __device__(float x) { return fmaxf(0.0, x); });
}

void relu_backward(Array *input_grad, const Array *output_grad,
                   const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "relu_backward: size mismatch between input grad and output grad");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "relu_backward: size mismatch between input and its grad");

    thrust::transform(
        input->get_vec().begin(), input->get_vec().end(),
        output_grad->get_vec().begin(), input_grad->get_vec().begin(),
        [] __device__(float x, float g) { return x > EPS ? g : 0; });
}

void ReLU::forward() {
    Array *input = prev->get_output();
    relu_forward(input, input);
}

void ReLU::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    relu_backward(output_grad, output_grad, input);
}

void sigmoid_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "sigmoid_forward: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [] __device__(float x) { return 1 / (1 + expf(-x)); });
}

void sigmoid_backward(Array *input_grad, const Array *output_grad,
                      const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "sigmoid_backward: size mismatch between input grad and output "
             "grad");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "sigmoid_backward: size mismatch betwen input and its grad");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output_grad->get_vec().begin(),
                      input_grad->get_vec().begin(),
                      [] __device__(float x, float g) {
                          float sigmoid = 1 / (1 + expf(-x));
                          return g * sigmoid * (1 - sigmoid);
                      });
}

void Sigmoid::forward() {
    Array *input = prev->get_output();
    sigmoid_forward(input, input);
}

void Sigmoid::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    sigmoid_backward(output_grad, output_grad, input);
}

void tanh_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "tanh_forward: size mismatch between input and output");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output->get_vec().begin(),
                      [] __device__(float x) { return tanhf(x); });
}

void tanh_backward(Array *input_grad, const Array *output_grad,
                   const Array *input) {
    CHECK_EQ(input_grad->get_vec().size(), output_grad->get_vec().size(),
             "tanh_backward: size mismatch between input grad and output grad");
    CHECK_EQ(input_grad->get_vec().size(), input->get_vec().size(),
             "tanh_backward: size mismatch between input and its grad");

    thrust::transform(input->get_vec().begin(), input->get_vec().end(),
                      output_grad->get_vec().begin(),
                      input_grad->get_vec().begin(),
                      [] __device__(float x, float g) {
                          float tanh = tanhf(x);
                          return g * (1 - tanh * tanh);
                      });
}

void Tanh::forward() {
    Array *input = prev->get_output();
    tanh_forward(input, input);
}

void Tanh::backward() {
    const Array *input = prev->get_output();
    Array *output_grad = next->get_grad();
    tanh_backward(output_grad, output_grad, input);
}

// Applies softmax function to a batch of vectors so that elements in each
// output vector lie in the range (0, 1) and sum to 1.
// Softmax function is defined as: Softmax(X) = exp(X) / sum(exp(X)).
// This implementation rescales the input vectors by subtracting the maxinum
// value from every vector element to avoid overflow when calling exp().
__global__ void softmax_forward_kernel(int size, float *output,
                                       const float *input, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input += idx * stride;
        output += idx * stride;

        float max_val = -FLT_MAX;
        for (int i = 0; i < stride; i++) {
            max_val = fmaxf(max_val, input[i]);
        }

        float exp_sum = 0;
        for (int i = 0; i < stride; i++) {
            output[i] = expf(input[i] - max_val);
            exp_sum += output[i];
        }

        for (int i = 0; i < stride; i++) {
            // Add episilon to the output to prevent applying log on 0 when
            // calculating loss.
            output[i] = output[i] / exp_sum + EPS;
        }
    }
}

void softmax_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_shape().size(), 2,
             "softmax_forward: output is not 2 dimensional");
    CHECK_EQ(input->get_shape().size(), 2,
             "softmax_forward: input is not 2 dimensional");
    CHECK_EQ(output->get_vec().size(), input->get_vec().size(),
             "softmax_forward: shape mismatch between input and output");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    int grid_size = ceil((float)batch_size / BLOCK_SIZE);

    softmax_forward_kernel<<<grid_size, BLOCK_SIZE>>>(batch_size, output_raw,
                                                      input_raw, batch_stride);
    CUDA_POST_KERNEL_CHECK;
}

void Softmax::forward() {
    const Array *input = prev->get_output();
    utils::set_array_ptr(output, input->get_shape());
    softmax_forward(output.get(), input);
}

void Softmax::backward() {
    const Array *output_grad = next->get_grad();
    utils::set_array_ptr(grad, output_grad->get_shape());

    // Backpropagation at Softmax layer involves only copying the output
    // gradient to the input gradient.
    thrust::copy(output_grad->get_vec().begin(), output_grad->get_vec().end(),
                 grad->get_vec().begin());
}

// Applies LogSoftmax function to a batch of vectors. LogSoftmax is defined as:
// LogSoftmax(X) = log(Softmax(X)) = log(exp(X) / sum(exp(X)))
//               = X - log(sum(exp(X)))
// This implementation rescales the input vectors by subtracting the maxinum
// value from every vector element to avoid overflow when calling exp().
__global__ void log_softmax_forward_kernel(int size, float *output,
                                           const float *input, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input += idx * stride;
        output += idx * stride;

        float max_val = -FLT_MAX;
        for (int i = 0; i < stride; i++) {
            max_val = fmaxf(max_val, input[i]);
        }

        float log_sum = 0;
        for (int i = 0; i < stride; i++) {
            log_sum += expf(input[i] - max_val);
        }
        log_sum = max_val + logf(log_sum);

        for (int i = 0; i < stride; i++) {
            output[i] = input[i] - log_sum;
        }
    }
}

void log_softmax_forward(Array *output, const Array *input) {
    CHECK_EQ(output->get_shape().size(), 2,
             "log_softmax_forward: output is not 2 dimensional");
    CHECK_EQ(input->get_shape().size(), 2,
             "log_softmax_forward: input is not 2 dimensional");
    CHECK_EQ(output->get_shape(), input->get_shape(),
             "log_softmax_forward: shape mismatch between input and output");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    const float *input_raw = RAW_PTR(input->get_vec());
    float *output_raw = RAW_PTR(output->get_vec());

    int grid_size = ceil((float)batch_size / BLOCK_SIZE);

    log_softmax_forward_kernel<<<grid_size, BLOCK_SIZE>>>(
        batch_size, output_raw, input_raw, batch_stride);

    CUDA_POST_KERNEL_CHECK;
}

// Calculates loss gradient w.r.t. input of LogSoftmax layer using the formula
// dL/dX = dL/dY - sum(dL/dY) * Softmax(X)
__global__ void log_softmax_backward_kernel(int size, float *input_grad,
                                            const float *output_grad,
                                            const float *input, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input_grad += idx * stride;
        output_grad += idx * stride;
        input += idx * stride;

        float max_val = -FLT_MAX;
        for (int i = 0; i < stride; i++) {
            max_val = fmaxf(max_val, input[i]);
        }

        float log_sum = 0;
        for (int i = 0; i < stride; i++) {
            log_sum += expf(input[i] - max_val);
        }
        log_sum = max_val + logf(log_sum);

        float dldy_sum = 0;
        for (int i = 0; i < stride; i++) {
            dldy_sum += output_grad[i];
        }

        for (int i = 0; i < stride; i++) {
            input_grad[i] =
                output_grad[i] - dldy_sum * expf(input[i] - log_sum);
        }
    }
}

void log_softmax_backward(Array *input_grad, const Array *output_grad,
                          const Array *input) {
    CHECK_EQ(input_grad->get_shape().size(), 2,
             "log_softmax_backward: input_grad is not 2 dimensional");
    CHECK_EQ(output_grad->get_shape().size(), 2,
             "log_softmax_backward: output_grad is not 2 dimensional");
    CHECK_EQ(input->get_shape().size(), 2,
             "log_softmax_backward: input is not 2 dimensional");

    CHECK_EQ(input_grad->get_shape(), output_grad->get_shape(),
             "log_softmax_backward: shape mismatch between output grad and "
             "input grad");
    CHECK_EQ(input_grad->get_shape(), input->get_shape(),
             "log_softmax_backward: shape mismatch between input and its grad");

    int batch_size = input->get_shape()[0];
    int batch_stride =
        std::accumulate(input->get_shape().begin() + 1,
                        input->get_shape().end(), 1, std::multiplies<int>());

    float *input_grad_raw = RAW_PTR(input_grad->get_vec());
    const float *output_grad_raw = RAW_PTR(output_grad->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    int grid_size = ceil((float)batch_size / BLOCK_SIZE);

    log_softmax_backward_kernel<<<grid_size, BLOCK_SIZE>>>(
        batch_size, input_grad_raw, output_grad_raw, input_raw, batch_stride);

    CUDA_POST_KERNEL_CHECK;
}

void LogSoftmax::forward() {
    const Array *input = prev->get_output();
    utils::set_array_ptr(output, input->get_shape());

    log_softmax_forward(output.get(), input);
}

void LogSoftmax::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();
    utils::set_array_ptr(grad, input->get_shape());

    log_softmax_backward(grad.get(), output_grad, input);
}

} // namespace nnv2