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
    void train_epoch();

    std::pair<int, int> top1_accuracy(const Array *preds, const Array *y);

    std::vector<std::unique_ptr<Layer>> layers;

    DataLoader *loader;
    Loss *loss;
    Optimizer *optimizer;

    thrust::device_vector<int> is_accurate;
};

} // namespace nnv2