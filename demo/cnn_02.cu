// Network based on LeNet-5 architecture (~62k parameters)
// The original description uses Sigmoid as the hidden activation layer but it
// doesn't learn quite well so I pick ReLU instead.

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
    // Computes 6 features using a 5x5 filter with ReLU activation.
    // Padding of (2, 2) is added to preserve width and height.
    // Input shape: [batch_size, 1, 28, 28]
    // Output shape: [batch_size, 6, 28, 28]
    // No. of params: 1*6*5*5 + 6 = 156
    net.add(new Conv2D(1, 6, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // First max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 6, 28, 28]
    // Output shape: [batch_size, 6, 14, 14]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Second convolution layer
    // Computes 16 features using a 5x5 filter with ReLU activation.
    // Input shape: [batch_size, 6, 14, 14]
    // Output shape: [batch_size, 16, 10, 10]
    // No. of params: 6*16*5*5 + 16 = 2416
    net.add(new Conv2D(6, 16, 14, 14, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // Second max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 16, 10, 10]
    // Output shape: [batch_size, 16, 5, 5]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Flatten inference batches from 2D image features to 1D neurons
    // Input shape: [batch_size, 16, 5, 5]
    // Output shape: [batch_size, 16*5*5]
    net.add(new Flatten);

    // First fully connected layer with 120 neurons
    // Input shape: [batch_size, 16*5*5]
    // Output shape: [batch_size, 120]
    // No. of params: 16*5*5*120 + 120 = 48_120
    net.add(new Linear(16 * 5 * 5, 120, init.get()));
    net.add(new ReLU);

    // Second fully connected layer with 84 neurons
    // Input shape: [batch_size, 120]
    // Output shape: [batch_size, 84]
    // No. of params: 120*84 + 84 = 10_164
    net.add(new Linear(120, 84, init.get()));
    net.add(new ReLU);

    // Logits layer
    // Input shape: [batch_size, 84]
    // Output shape: [batch_size, 10]
    // No. of params: 84*10 + 10 = 850
    net.add(new Linear(84, 10, init.get()));
    net.add(new Softmax);

    std::cout << "Network setup complete" << std::endl;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path), 32);
    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<SGD>(0.001);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(30, true);

    return 0;
}