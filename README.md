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
git clone --b main --single-branch https://github.com/ceilight/cudaranai.git
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

| Classifier | Optimizer | Run time | Max. test accuracy | Avg. test accuracy of the last 5 epochs |
| --- | --- | --- | --- | --- |
| 2 Conv + 1 FC (~11k params) | Adam | 76s | 0.9042 | 0.8998 |
| 2 Conv + 3 FC (~62k params) | SGD | 105s | 0.9031 | 0.9003 |
| 3 Conv + 2 FC (~193k params) | RMSProp | 138s | 0.9023 | 0.8986 |
| 3 Conv + 2 FC (~193k params) + Dropout | RMSProp | 142s | 0.9044 | 0.9004 |
| 3 Conv + 3 FC (~644k params) + Dropout | Adam | 322s | 0.9093 | 0.9062 |
| 2 Conv + 2 FC (~3.27M params) + Dropout | SGD | 265s | 0.9201 | 0.9161 |

Refer to this [notebook](https://colab.research.google.com/drive/1PTEictwtufbPmYrmPti-UT2d56daOq6B?usp=sharing)
for full demonstration of the training process.