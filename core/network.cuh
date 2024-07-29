#pragma once

#include "common.h"
#include "dataloader.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"

#include <memory>
#include <utility>
#include <vector>

namespace nnv2 {

class Network {
public:
    void add(Layer *layer);

    void init(DataLoader *loader, Loss *loss, Optimizer *optimizer);

    void train(int epochs, bool shuffle = false);
    void test();

private:
    void train_epoch();

    std::pair<int, int> top1_accuracy(const Array *preds, const Array *results);

    std::vector<std::unique_ptr<Layer>> layers;

    DataLoader *loader;
    Loss *loss;
    Optimizer *optimizer;
};

} // namespace nnv2