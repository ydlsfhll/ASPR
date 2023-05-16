# A Mathematics Framework of Artificial Shift Population Risk and Its Further Understanding Related to Consistent Regularization



## Introduction:

We propose a framework for shifted population risk and a consistency regularization strategy based on such connection. Experiments on CIFAR10/100, Food101 and Imagenet demonstrated that our methods outperform the standard training strategy in final error rate as well as convergence stability with the same data augmentation and computational cost.



## Setup

- NVIDIA GPU (NVIDIA RTX A6000 and NVIDIA A40 are used on ImageNet. NVIDIA GeForce RTX 3090 are used on other datasets)
- torch 2.0.0,  torchvision 0.15.1



## Running

Codes for CIFAR-10/100 and Food101 are placed in the folder named 'CIFAR and Food'. After you have cloned the repository, you can train each dataset of either CIFAR-10/100, Food101 by running the scripts below.

For baseline: 

```python
python train.py -d [CIFAR10/CIFAR100/Food101] -m [resnet50/resnet18/WideResNet-40-2/WideResNet-28-10] -mm -bl -lr 0.1 -bs [100/128] -we 10 -lrs cosine -s
```

For Ours: 

```python
python train.py -d [CIFAR10/CIFAR100/Food101] -m [resnet50/resnet18/WideResNet-40-2/WideResNet-28-10] -mm -lr 0.1 -bs [100/128] -we 10 -lrs cosine -s -b 0.5
```



Codes for ImageNet are placed in the folder named 'ImageNet'. To accelerate the training process, we conducted separate training and validation and adopt mixed precision training with torch.cuda.amp. Similar to the CIFAR and Food101, you need to added '-bl' for the baseline experiments, while our proposed strategy was run without it. The following commands are for our strategy.

For train:

```python
python train_only.py -d ImageNet -n 11 -pf 11 -m [resnet50\resnet101] -lr [0.1/0.075] -lrs milestone -bs [256/192] -we 10 -a common_top_5 -b 0.5 -s
```

For test:

```python
python evaluate_only.py -d ImageNet -n 10 -pf 10 -m [resnet50\resnet101] -lrs milestone -bs 256 -we 10 -a common_top_5 -b 0.5 -e 200 -s
```



## References & Opensources

- **Wide-ResNet** : [paper](https://arxiv.org/pdf/1605.07146), [code](https://github.com/meliketoy/wide-resnet.pytorch)

