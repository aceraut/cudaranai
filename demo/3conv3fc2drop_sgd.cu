// https://github.com/umbertogriffo/Fashion-mnist-cnn-keras/blob/master/src/convolutional/fashion_mnist_cnn.py

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
    std::unique_ptr<Initializer> init = std::make_unique<XavierUniform>();

    net.add(new Conv2D(1, 32, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    net.add(new Conv2D(32, 64, 14, 14, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    net.add(new Conv2D(64, 128, 7, 7, 1, 1, 1, 1, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    net.add(new Flatten);

    net.add(new Linear(128 * 4 * 4, 1024, init.get()));
    net.add(new ReLU);
    net.add(new Dropout(0.5));

    net.add(new Linear(1024, 512, init.get()));
    net.add(new ReLU);
    net.add(new Dropout(0.5));

    net.add(new Linear(512, 10, init.get()));
    net.add(new Softmax);

    std::cout << "Network setup complete" << std::endl;

    int epochs = 30;
    float lr = 0.01;
    float decay = lr / 150;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path), 32);
    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<SGD>(lr, decay, 0.9);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(epochs, true);

    return 0;
}