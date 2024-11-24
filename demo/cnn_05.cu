// https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
// 2 convolution + 2 fully connected layers (~3.27M params)

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

    // First convolution layer
    // Computes 32 features using a 5x5 filter with ReLU activation.
    // Padding of (2, 2) is added to preserve width and height.
    // Input shape: [batch_size, 1, 28, 28]
    // Output shape: [batch_size, 32, 28, 28]
    // No. of params: 1*32*5*5 + 32 = 832
    net.add(new Conv2D(1, 32, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // First max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 32, 28, 28]
    // Output shape: [batch_size, 32, 14, 14]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Second convolution layer
    // Computes 64 features using a 5x5 filter with ReLU activation.
    // Padding of (2, 2) is added to preserve width and height.
    // Input shape: [batch_size, 32, 14, 14]
    // Output shape: [batch_size, 64, 14, 14]
    // No. of params: 32*64*5*5 + 64 = 51_264 
    net.add(new Conv2D(32, 64, 14, 14, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // Second max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 64, 14, 14]
    // Output shape: [batch_size, 64, 7, 7]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Flatten inference batches from 2D image features to 1D neurons
    // Input shape: [batch_size, 64, 7, 7]
    // Output shape: [batch_size, 64*7*7]
    net.add(new Flatten);

    // Fully connected layer with 1024 neurons
    // Input shape: [batch_size, 64*7*7]
    // Output shape: [batch_size, 1024]
    // No. of params: 64*7*7*1024 + 1024 = 3_212_288
    net.add(new Linear(64 * 7 * 7, 1024, init.get()));
    net.add(new ReLU);

    // Dropout layer
    // Randomly zeroes out 40% of the current neurons
    net.add(new Dropout(0.4));

    // Logits layer
    // Input shape: [batch_size, 1024]
    // Output shape: [batch_size, 10]
    // No. of params: 1024*10 + 10 = 10_250
    net.add(new Linear(1024, 10, init.get()));
    net.add(new Softmax);

    std::cout << "Network setup complete" << std::endl;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path), 400);
    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<SGD>(0.001);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(30, true);

    return 0;
}