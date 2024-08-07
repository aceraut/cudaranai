Convolutional neural network from scratch (well it does have dependencies but
technically they're pre-installed) in CUDA/C++. The implementation is not
blazingly fast, however, because of my skill issue in CUDA.

The name is a pun on kudaranai, which means useless in Japanese, and I think
it's the fittest description of this repo.

Don't mind the crappy commit history, I just don't have CUDA-enabled hardware
to test locally.

## How to build

GPU hardware that supports CUDA compiler (nvcc) version 12 or above is required.
Other dependencies include Thrust and cuRAND libraries (which are probably
shipped along with the compiler).

```bash
git clone --branch main https://github.com/ceilight/cudaranai
cd cudaranai
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Demo

Table below documents the result of training different classifiers on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset after
30 epochs. Refer to this [notebook](https://colab.research.google.com/drive/1PTEictwtufbPmYrmPti-UT2d56daOq6B?usp=sharing)
for full demonstration of the training process.

| Classifier | Parameters | Optimizer | Run time | Max. test accuracy |
| --- | --- | --- | --- | --- |
| 2 Conv + 3 FC | ~62k | SGD | 6 min | 0.9049 |
| 2 Conv + 3 FC | ~62k | RMSProp | 6 min | 0.902 |
| 2 Conv + 1 FC | ~11k | Adam | 5 min | 0.9088 |
| 3 Conv + 2 FC | ~193k | RMSProp | 6 min | 0.8961 |