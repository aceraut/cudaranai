[nnv2](https://github.com/ceilight/nnv2) but the core library is rewritten in
CUDA. The implementation isn't blazingly fast as I'm only new to CUDA
programming.

The name is a pun on kudaranai, which means useless in Japanese, and I think
it's the fittest description of this repo.

Don't mind the crappy commit history, I just don't have CUDA-enabled hardware
to test locally.

## Results

Table below documents the result of training different classifiers on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset after
30 epochs.

| Classifier | Parameters | Optimizer | Run time | Max. test accuracy |
| --- | --- | --- | --- | --- |
| 2 Conv + 1 FC | ~11k | Adam | 5 min | 0.9088 |
| 2 Conv + 3 FC | ~62k | SGD | 6 min | 0.9049 |
| 2 Conv + 3 FC | ~62k | RMSProp | 6 min | 0.902 |
| 3 Conv + 2 FC | ~193k | RMSProp | 6 min | 0.8961 |

Refer to this [notebook](https://colab.research.google.com/drive/1PTEictwtufbPmYrmPti-UT2d56daOq6B?usp=sharing)
for full demonstration of the training process.