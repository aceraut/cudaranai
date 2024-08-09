An **unoptimized** implementation of convolutional neural network, made from
scratch (well technically it does have dependencies but they're pre-installed)
in CUDA/C++.

Don't mind the crappy commit history, I just don't have CUDA-enabled hardware
to test locally.

## How to build

GPU hardware that supports CUDA compiler (nvcc) version 12 or above is required.
Other dependencies include Thrust and cuRAND libraries (which are probably
shipped along with the compiler).

```bash
git clone --branch main https://github.com/ceilight/cudaranai.git
cd cudaranai
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Demo

Table below documents the result of training different classifiers on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset after
30 epochs. Test accuracy may differ due to batch size, random seed for weight
initialization, etc.

| Classifier | Preprocessing | Optimizer | Run time | Max. test accuracy |
| --- | --- | --- | --- | --- |
| 2 Conv + 1 FC ~11k params | Standardization | Adam | 70 sec | 0.9078 |
| 2 Conv + 3 FC ~62k params | Standardization | SGD | 90 sec | 0.9071 |
| 2 Conv + 3 FC ~62k params | Standardization | RMSProp | 65 sec | 0.9024 |
| 3 Conv + 2 FC ~193k params | Scaling | RMSProp | 152 sec | 0.9074 |

Refer to this [notebook](https://colab.research.google.com/drive/1L7bxc2k-IakPnZemFQqPm9VdVAtycabU?usp=sharing)
for full demonstration of the training process.