import os
import random

from torchvision import datasets
import torchvision.transforms as T
import torch
from torch.nn import Identity
from torch.utils.data import Dataset, Subset


dataset_dir = os.path.dirname(os.path.abspath(__file__)) + r'/dataset'

def get_dataset(args):
    def complementary_list(U, A):
        return list(set(U).difference(set(A)))

    if args.dataset == 'CIFAR10':
        transform_train = T.Compose([T.Resize((224, 224)),
                                    T.RandomHorizontalFlip() if args.baseline else Identity(),
                                    T.RandomCrop(224, 28) if args.baseline else Identity(), 
                                    T.RandomRotation(15) if args.baseline else Identity(), 
                                    T.ToTensor()])
        transform_test = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor()])
        
        dataset_train = datasets.CIFAR10(root = dataset_dir, transform = transform_train, train = True)
        dataset_test = datasets.CIFAR10(root = dataset_dir, transform = transform_test, train = False)
            
        n = len(dataset_train)
        i_val = [[] for _ in range(0, 100)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(args.vaild_ratio * n / 100)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        dataset_valid = Subset(dataset_train, i_val)
        
        dataset_train = datasets.CIFAR10(root = dataset_dir, transform = transform_train, train = True)
        dataset_train = Subset(dataset_train, complementary_list(range(0, n), i_val))
        
        dataset_test = datasets.CIFAR10(root = dataset_dir, transform = transform_test, train = False)
    
    elif args.dataset == 'CIFAR100':
        transform_train = T.Compose([T.Resize((224, 224)),
                                    T.RandomHorizontalFlip() if args.baseline else Identity(),
                                    T.RandomCrop(224, 28) if args.baseline else Identity(), 
                                    T.RandomRotation(15) if args.baseline else Identity(), 
                                    T.ToTensor()])
        transform_test = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor()])
        
        dataset_train = datasets.CIFAR100(root = dataset_dir, transform = transform_train, train = True)
        # print(len(dataset_train))

        n = len(dataset_train)
        i_val = [[] for _ in range(0, 100)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(args.vaild_ratio * n / 100)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        dataset_valid = Subset(dataset_train, i_val)
        
        dataset_train = datasets.CIFAR100(root = dataset_dir, transform = transform_train, train = True)
        dataset_train = Subset(dataset_train, complementary_list(range(0, n), i_val))
        
        dataset_test = datasets.CIFAR100(root = dataset_dir, transform = transform_test, train = False)

    elif args.dataset == "ImageNet":
        TRAIN_MEAN = [0.485, 0.456, 0.406] 
        TRAIN_STD = [0.229, 0.224, 0.225] 
        temp_dataset_dir = r'../GCLmaster/data/ImageNet/'
        transform_train = T.Compose([T.Resize((224, 224)),
                                    T.RandomHorizontalFlip() if args.baseline else Identity(),
                                    T.RandomCrop(224, 28) if args.baseline else Identity(), 
                                    T.RandomRotation(15) if args.baseline else Identity(), 
                                    T.ToTensor(),
                                    T.Normalize(TRAIN_MEAN, TRAIN_STD)])
        transform_test = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(TRAIN_MEAN, TRAIN_STD)])
        dataset_train = datasets.ImageFolder(root = temp_dataset_dir + r'train', transform = transform_train)
        dataset_valid = datasets.ImageFolder(root = temp_dataset_dir + r'val', transform = transform_test)
        # dataset_train = Subset(dataset_train, list(range(0, 5120)))
        # dataset_valid = Subset(dataset_valid, list(range(0, 5120)))
        dataset_test = None
        print(len(dataset_train))
        print(len(dataset_valid))

    return dataset_train, dataset_valid, dataset_test