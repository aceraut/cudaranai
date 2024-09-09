#pragma once

#include "common.cuh"
#include "dataloader.cuh"
#include "layer.cuh"
#include "loss.cuh"
#include "optimizer.cuh"

#include <memory>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>

namespace nnv2 {

class Network {
public:
    void add(Layer *layer);

    void init(DataLoader *loader, Loss *loss, Optimizer *optimizer);

    void train(int epochs, bool shuffle = false);
    void test();

private:
    // A single run of training classifiers.
    void train_epoch();

    // Calculates top1 accuracy of prediction compared to actual result y
    std::pair<int, int> top1_accuracy(const Array *preds, const Array *y);

    std::vector<std::unique_ptr<Layer>> layers;

    DataLoader *loader;
    Loss *loss;
    Optimizer *optimizer;

    // Each element denotes whether an input in a batch predicts correctly.
    thrust::device_vector<int> is_accurate;
};

} // namespace nnv2