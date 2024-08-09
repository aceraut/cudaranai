// This file implements the Network class, representing the neural network
//
// Pretty much a lot of stuffs are involved, but a Network object requires
// layers for constructing the neural network, a DataLoader that acts as
// the data provider, the Loss object, an optimizer to update the parameters.

#include "common.cuh"
#include "network.cuh"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

#ifdef STATS
#include <ctime>
#endif

#include <thrust/extrema.h>

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

#ifdef STATS
    forward_time.resize(layers.size());
    backward_time.resize(layers.size());
#endif
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
    int accurate_count = 0;
    int sample_count = 0;

#ifdef STATS
    std::clock_t start;

    load_time = 0;
    verify_time = 0;
    optim_time = 0;
    std::fill(forward_time.begin(), forward_time.end(), 0);
    std::fill(backward_time.begin(), backward_time.end(), 0);
#endif

    while (loader->has_next_train_batch()) {
        batch_count++;

#ifdef STATS
        start = std::clock();
#endif
        loader->load_train_batch();
#ifdef STATS
        load_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif

        // perform forward propagation to calculate prediction
        for (int i = 0; i < layers.size(); i++) {
#ifdef STATS
            start = std::clock();
#endif
            layers[i]->forward();
#ifdef STATS
            forward_time[i] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif
        }

        // calculate loss & accuracy of prediction compared to actual result
#ifdef STATS
        start = std::clock();
#endif
        loss_sum += loss->calculate_loss(loader->get_labels());
        std::pair<int, int> accuracy =
            top1_accuracy(layers.back()->get_output(), loader->get_labels());
        accurate_count += accuracy.first;
        sample_count += accuracy.second;

        // backpropagate the loss gradient to the layers
        loss->backward();
#ifdef STATS
        verify_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif

        for (int i = layers.size() - 1; i >= 0; i--) {
#ifdef STATS
            start = std::clock();
#endif
            layers[i]->backward();
#ifdef STATS
            backward_time[i] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif
        }
        // update the parameters with regard to the gradient
#ifdef STATS
        start = std::clock();
#endif
        optimizer->update_parameters();
#ifdef STATS
        optim_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif
    }

    std::cout << "Avg loss: " << loss_sum / batch_count << ", ";
    std::cout << "Avg accuracy: " << 1.0 * accurate_count / sample_count
              << "; ";

#ifdef STATS
    std::cout << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Load: " << load_time / batch_count << std::endl;
    for (int i = 0; i < (int)forward_time.size(); i++)
        std::cout << "Forward #" << i << ": " << forward_time[i] / batch_count
                  << std::endl;
    std::cout << "Verify: " << verify_time / batch_count << std::endl;
    for (int i = 0; i < (int)backward_time.size(); i++)
        std::cout << "Backward #" << i << ": " << backward_time[i] / batch_count
                  << std::endl;
    std::cout << "Optimize: " << optim_time << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

#endif
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
std::pair<int, int> Network::top1_accuracy(const Array *preds, const Array *y) {
    int batch_size = preds->get_shape()[0];
    int labels = preds->get_shape()[1];
    int count = 0;

    for (int k = 0; k < batch_size; k++) {
        // find label with highest distribution in prediction batch
        auto pred_start = preds->get_vec().begin() + k * labels;
        auto pred_top1 = thrust::max_element(pred_start, pred_start + labels);
        int pred_top1_idx = (int)(pred_top1 - pred_start);

        // do the same with actual result batch
        auto result_start = y->get_vec().begin() + k * labels;
        auto result_top1 =
            thrust::max_element(result_start, result_start + labels);
        int result_top1_idx = (int)(result_top1 - result_start);

        if (pred_top1_idx == result_top1_idx) {
            count++;
        }
    }
    return std::make_pair(count, batch_size);
}

} // namespace nnv2