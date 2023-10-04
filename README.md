# A Mathematics Framework of Artificial Shift Population Risk and Its Further Understanding Related to Consistent Regularization



## Introduction:

We propose a framework for shifted population risk and a consistency regularization strategy based on such connection. Experiments on CIFAR10/100, Food101 and Imagenet demonstrated that our methods outperform the standard training strategy in final error rate as well as convergence stability with the same data augmentation and computational cost.



## Setup

- NVIDIA GPU (NVIDIA RTX A6000 and NVIDIA A40 are used on ImageNet. NVIDIA GeForce RTX 3090 are used on other datasets)
- torch 2.0.0,  torchvision 0.15.1



## Running

### Standard Scenario Experiment

Codes for CIFAR-10/100 and Food101 are placed in the folder named 'CIFAR and Food'. After you have cloned the repository, you can train each dataset of either CIFAR-10/100, Food101 by running the scripts below.

For baseline: 

```python
python train.py -d [CIFAR10/CIFAR100/Food101] -m [resnet50/resnet18/WideResNet-40-2/WideResNet-28-10] -mm -bl -lr 0.1 -bs [100/128] -we [5/10] -lrs cosine -s
```

For Ours (-b is used to choose the coefficients of our proposed regularization): 

```python
python train.py -d [CIFAR10/CIFAR100/Food101] -m [resnet50/resnet18/WideResNet-40-2/WideResNet-28-10] -mm -lr 0.1 -bs [100/128] -we [5/10] -lrs cosine -s -b 0.5
```



Codes for ImageNet are placed in the folder named 'ImageNet'. To accelerate the training process, we conducted separate training and validation and adopt mixed precision training with torch.cuda.amp. Similar to the CIFAR and Food101, you need to add '-bl' for the baseline experiments, while our proposed strategy was run without it. The following commands are for our strategy.

For train:

```python
python train_only.py -d ImageNet -n 11 -pf 11 -m [resnet50\resnet101] -lr [0.1/0.075] -lrs milestone -bs [256/192] -we 10 -a common_top_5 -b 0.5 -s
```

For test:

```python
python evaluate_only.py -d ImageNet -n 10 -pf 10 -m [resnet50\resnet101] -lrs milestone -bs 256 -we 10 -a common_top_5 -b 0.5 -e 200 -s
```



### Long-Tailed Scenario Experiment

Codes are placed in the folder named 'CIFAR and Food'. In contrast to Standard Scenario Experiment, you need to change the name of the dataset in the command. You can use '-i' to change the value of  imbalance ratio. We pre-stored the validation set and test set using "torch.save" and read them directly before training. 

For baseline:

```
python train.py -d lt-CIFAR10 -m resnet18 -mm -bl -lr 0.1 -bs 100 -we 5 -lrs cosine -s -i 10 -a auc_ap_acc
```

For Ours:

```
python train.py -d lt-CIFAR10 -m resnet18 -mm -b 0.5 -lr 0.1 -bs 100 -we 5 -lrs cosine -s -i 10 -a auc_ap_acc
```



### OOD Scenario Experiment

The scripts in this folder provides PyTorch implementation of OOD scenario experiment in our paper. Before training, one should move datasets in `./datasets` or just modify the value of `dataset_dir` in `datasets.py`.

To start a experiment of ERM algorithm:

```
python train.py -al ERM
```

To start a experiment of our proposed algorithm:

```
python train.py -al ours
```

To modify hyperparameters, one can see the command below or just check `configurations.py`:

```
python train.py -d dataset -o optimizer -e train_epochs\
                -lr learning_rate -bs batch_size -b beta\
                -seed seed -lns log_name_suffix
```



## References & Opensources

- **Wide-ResNet** : [paper](https://arxiv.org/pdf/1605.07146), [code](https://github.com/meliketoy/wide-resnet.pytorch)

