// LeNet-5 architecture example
// The original description uses Sigmoid as the hidden activation layer but it's
// not learning quite well so I opt for ReLU

// Epoch: 30, Batch size: 32
// Init: LeCun Uniform, Loss: Softmax-CrossEntropy, Optimizer: SGD

#include "nnv2.cuh"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    using namespace nnv2;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];

    Network net;
    std::unique_ptr<Initializer> init = std::make_unique<LecunUniform>();

    net.add(new Conv2D(1, 6, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));
    net.add(new Conv2D(6, 16, 14, 14, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));
    net.add(new Flatten);
    net.add(new Linear(400, 120, init.get()));
    net.add(new ReLU);
    net.add(new Linear(120, 84, init.get()));
    net.add(new ReLU);
    net.add(new Linear(84, 10, init.get()));
    net.add(new Softmax);

    std::cout << "Network setup complete" << std::endl;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path, true), 32);
    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<SGD>(0.001);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(30, true);

    return 0;
}