// This file implements the Network class, representing the neural network
//
// Pretty much a lot of stuffs are involved, but a Network object requires
// layers for constructing the neural network, a DataLoader that acts as
// the data provider, the Loss object, an optimizer to update the parameters.

#include "common.cuh"
#include "network.cuh"

#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <utility>

#include <thrust/reduce.h>

namespace nnv2 {

// Appends a layer to the network
void Network::add(Layer *layer) {
    std::unique_ptr<Layer> next(layer);
    layers.push_back(std::move(next));
}

// Introduces DataLoader, Loss and optimizer and connect them to the layers
void Network::init(DataLoader *loader_, Loss *loss_, Optimizer *optimizer_) {
    loader = loader_;
    loss = loss_;
    optimizer = optimizer_;

    CHECK_COND(layers.size() > 0, "No layers found in the network");

    // connect loader to the first layer
    loader->connect(layers.front().get());
    // connect each layer to the subsequent one
    for (int i = 1; i < layers.size(); i++) {
        layers[i - 1]->connect(layers[i].get());
    }
    // connect the last layer to loss layer
    layers.back()->connect(loss);

    // register parameters to the optimizer
    for (int i = 0; i < layers.size(); i++) {
        optimizer->add_parameters(layers[i]->get_parameters());
    }
}

void Network::train(int epochs, bool shuffle) {
    for (int e = 0; e < epochs; e++) {
        std::cout << "[Epoch: " << e + 1 << "/" << epochs << "] ";
        loader->reset(shuffle);
        train_epoch();
        test();
    }
}

void Network::train_epoch() {
    float loss_sum = 0.0;
    int batch_count = 0;

    while (loader->has_next_train_batch()) {
        batch_count++;
        loader->load_train_batch();

        // perform forward propagation to calculate prediction
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->forward();
        }

        // calculate loss value of prediction compared to actual result
        loss_sum += loss->calculate_loss(loader->get_labels());

        // backpropagate the loss gradient to the layers
        loss->backward();
        for (int i = layers.size() - 1; i >= 0; i--) {
            layers[i]->backward();
        }
        // update the parameters with regard to the gradient
        optimizer->update_parameters();
    }

    std::cout << "Avg loss (train): " << loss_sum / batch_count << "; ";
}

void Network::test() {
    float loss_sum = 0.0;
    int batch_count = 0;
    int accurate_count = 0;
    int sample_count = 0;

    while (loader->has_next_test_batch()) {
        batch_count++;
        loader->load_test_batch();

        // perform forward propagation to calculate prediction
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->forward();
        }

        // calculate loss & accuracy of prediction compared to actual result
        loss_sum += loss->calculate_loss(loader->get_labels());
        std::pair<int, int> accuracy =
            top1_accuracy(layers.back()->get_output(), loader->get_labels());
        accurate_count += accuracy.first;
        sample_count += accuracy.second;
    }

    // print some stats here
    std::cout << "Avg loss (test): " << loss_sum / batch_count << ", ";
    std::cout << "Avg accuracy (test): " << 1.0 * accurate_count / sample_count
              << std::endl;
}

// Calculate the accuracy where the label with the highest probability
// is the correct label

// TODO: optimize max reduce op in the kernel (and other similar reduce ops)
__global__ void top1_accuracy_kernel(int size, int *is_accurate,
                                     const float *preds, const float *y,
                                     int label_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        preds += idx * label_stride;
        y += idx * label_stride;

        float max_val = -FLT_MAX;
        int pred_label = -1;
        int y_label = -1;

        for (int i = 0; i < label_stride; i++) {
            if (max_val < preds[i]) {
                max_val = preds[i];
                pred_label = i;
            }
        }

        for (int i = 0; i < label_stride; i++) {
            if (y[i] == 1) {
                y_label = i;
                break;
            }
        }

        is_accurate[idx] = (pred_label == y_label ? 1 : 0);
    }
}

std::pair<int, int> Network::top1_accuracy(const Array *preds, const Array *y) {
    int batch_size = preds->get_shape()[0];
    int label_stride = preds->get_shape()[1];

    is_accurate.resize(batch_size);

    int *is_accurate_raw = RAW_PTR(is_accurate);
    const float *preds_raw = RAW_PTR(preds->get_vec());
    const float *y_raw = RAW_PTR(y->get_vec());

    int grid_size = ceil((float)batch_size / BLOCK_SIZE);

    top1_accuracy_kernel<<<grid_size, BLOCK_SIZE>>>(
        batch_size, is_accurate_raw, preds_raw, y_raw, label_stride);
    CUDA_POST_KERNEL_CHECK;

    int count = thrust::reduce(is_accurate.begin(), is_accurate.end());
    return std::make_pair(count, batch_size);
}

} // namespace nnv2