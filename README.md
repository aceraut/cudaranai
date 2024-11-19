An **unoptimized** implementation of convolutional neural network, made from
scratch (well technically it does have dependencies but they're pre-installed)
in CUDA/C++.

Don't mind the crappy commit history, I just don't have NVIDIA graphic cards
to test it locally.

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

| Classifier | Preprocessing | Optimizer | Run time | Max. test accuracy | Final test accuracy |
| --- | --- | --- | --- | --- | --- |
| 2 Conv + 1 FC (~11k params) | Standardization | Adam | 68 sec | 0.9061 | 0.9034 |
| 2 Conv + 3 FC (~62k params) | Standardization | SGD | 84 sec | 0.9068 | 0.8927 |
| 2 Conv + 3 FC (~62k params) | Standardization | RMSProp | 61 sec | 0.8968 | 0.8964 |
| 3 Conv + 2 FC (~193k params) | Scaling | RMSProp | 151 sec | 0.9023 | 0.9013 |
| 3 Conv + 2 FC + Dropout (~193k params) | Scaling | RMSProp | 155 sec | 0.9027 | 0.8997 |
| 3 Conv + 3 FC + Dropout (~685k params) | Standardization | Adam | 448 sec | 0.9022 | 0.9022 |

Refer to this [notebook](https://colab.research.google.com/drive/1PTEictwtufbPmYrmPti-UT2d56daOq6B?usp=sharing)
for full demonstration of the training process.