# nnv2

nnv2 is a toy implementation of a Convolutional Neural Network (CNN) from
scratch, completed as part of a set of programming challenges from /dpt/ thread.

> **Note:**
> This implementation only makes use of CPU computation so training model with
this is very slow. I might dab some CUDA on it to speed things up later on.

## Features

### Layers

- DataLoader (input layer)
- Linear
- Conv2D
- MaxPool2D
- Flatten
- Activation layers (ReLU, Tanh, Sigmoid, Softmax, LogSoftmax)

### Others

- Dataset driver (MNIST)
- Initializers (Kaiming, LeCun, Xavier)
- Loss (CrossEntropyLoss, NLLLoss)
- Optimizers (SGD, RMSProp, Adam)

## How to build

### Install dependencies

If you intend to build this on your own, you will need C++ compiler compatible
with C++17 or later, C++ standard library, OpenBLAS, and CMake 3.10 or later.

```bash
sudo apt install gcc g++ libopenblas-dev cmake
# Find directory that has OpenBLASConfig.cmake file so CMake can locate OpenBLAS
find /usr -iname 'OpenBLASConfig.cmake'
```

### Compile

```bash
git clone --branch main https://github.com/ceilight/nnv2.git
cd nnv2
mkdir build
cd build
cmake -DOpenBLAS_DIR=<openblas_cmake_config_directory> ..
make # or cmake --build .
```

### Run examples

Download and unzip [MNIST](https://yann.lecun.com/exdb/mnist/) or
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
dataset in a folder of your choice.

Train a model using a neural network (for example LeNet-5).

```bash
./lenet5
Usage: ./lenet5 <dataset_path>
```

Profit?

## Results

On [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset:

| Classifier | Parameters | Optimizer | Computation time (after 30 epochs) | Test accuracy |
| --- | --- | --- | --- | --- |
| 2 Conv + 1 FC | ~11k | Adam | 14 min | 0.8922 |
| 2 Conv + 3 FC | ~62k | SGD | 10 min | 0.8977 |
| 2 Conv + 3 FC | ~62k | RMSProp | 14 min | 0.8952 |
| 3 Conv + 2 FC | ~193k | RMSProp | 2 hr 5 min (lmao) | 0.8884 |

## References and related projects

- [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) - C++ demo of deep
neural networks