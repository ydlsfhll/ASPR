import os
import random
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import Subset
import torchvision
import torch


dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/dataset'

def get_dataset(args):
    def complementary_list(U, A):
        return list(set(U).difference(set(A)))

    if args.dataset == 'CIFAR100':
        mean_std = [[0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]]
    elif args.dataset == 'CIFAR10':
        mean_std = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]]
    
    if args.model_backbone == 'resnet18' or args.model_backbone == 'resnet50' or args.model_backbone == 'resnet32':
    
        transform_train = T.Compose([T.Resize((224, 224)),
                                        T.ToTensor()])
        transform_test = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor()])
    elif args.dataset == 'SVHN':
        transform_train = T.Compose([T.ToTensor()])
        transform_test = T.Compose([T.ToTensor()])
        
    else:
        transform_train = T.Compose([T.ToTensor(),
                                    T.Normalize(*mean_std)])
        transform_test = T.Compose([T.ToTensor(),
                                    T.Normalize(*mean_std)])
    
    if args.dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(
            root = dataset_dir, transform = transform_train, train = True)
        dataset_test = datasets.CIFAR10(
            root = dataset_dir, transform = transform_test, train = False)
            
        n = len(dataset_train)
        i_val = [[] for _ in range(0, 10)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(args.vaild_ratio * n / 10)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        
        dataset_vaild = Subset(dataset_train, i_val)
        dataset_train = Subset(dataset_train, complementary_list(range(0, n), i_val))
        
    elif args.dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(
            root = dataset_dir, transform = transform_train, train = True)
        dataset_test = datasets.CIFAR100(
            root = dataset_dir, transform = transform_test, train = False)

        n = len(dataset_train)
        i_val = [[] for _ in range(0, 100)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(args.vaild_ratio * n / 100)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        
        dataset_vaild = Subset(dataset_train, i_val)
        dataset_train = Subset(dataset_train, complementary_list(range(0, n), i_val))
        

    elif args.dataset == "Food101":
        dataset_train = datasets.Food101(root = '../GCLmaster/data',download=False,transform = transform_train, split='train')
        dataset_test = datasets.Food101(root = '../GCLmaster/data',download=False,transform = transform_test, split='test')
        
        n = len(dataset_train)
        i_val = [[] for _ in range(0, 101)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(0.1 * n / 101)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        dataset_vaild = Subset(dataset_train, i_val)
        dataset_train = Subset(dataset_train, list(set(range(0, n)).difference(set(i_val))))
        del i_val
        
    elif args.dataset == "SVHN":
        dataset_train = torchvision.datasets.SVHN(root = '../GCLmaster/data', split='train', transform = transform_train, download=False)
        dataset_test = torchvision.datasets.SVHN(root = '../GCLmaster/data', split='test', transform = transform_test, download=False)
        
        n = len(dataset_train)
        i_val = [[] for _ in range(0, 10)]
        for i, (_, label) in enumerate(dataset_train):
            i_val[label].append(i)
        n_each_class = int(0.1 * n / 10)
        i_val = [random.sample(indexes, n_each_class) for indexes in i_val]
        i_val = [row[i] for i in range(0, n_each_class) for row in i_val]
        dataset_vaild = torch.utils.data.Subset(dataset_train, i_val)
        dataset_train = torch.utils.data.Subset(dataset_train, list(set(range(0, n)).difference(set(i_val))))
        del i_val
        
    elif args.dataset == 'lt-CIFAR10':
        dataset_train = datasets.CIFAR10(
            root = dataset_dir, transform = transform_train, train = True)
        dataset_test = datasets.CIFAR10(
            root = dataset_dir, transform = transform_test, train = False)
            
        n = len(dataset_train)
        i_train_val = [[] for _ in range(0, 10)]
        for i, (_, label) in enumerate(dataset_train):
            i_train_val[label].append(i)
        n_each_class = [int((n / 10) * (args.vaild_ratio + (1 - args.vaild_ratio) * (1 / args.imbalance) ** (t / 9))) for t in range(0, 10)]
        i_train_val = [random.sample(indexes, n_each_class[i]) for i, indexes in enumerate(i_train_val)]
        i_val = [random.sample(indexes, int(args.vaild_ratio * n / 10)) for i, indexes in enumerate(i_train_val)]
        
        i_train_val, i_val = sum(i_train_val, []), sum(i_val, [])
        dataset_vaild = Subset(dataset_train, i_val)
        print('vl:',len(dataset_vaild))
        dataset_train = Subset(dataset_train, complementary_list(i_train_val, i_val))

    elif args.dataset == 'lt-CIFAR100':
        dataset_train = datasets.CIFAR100(
            root = dataset_dir, transform = transform_train, train = True)
        dataset_test = datasets.CIFAR100(
            root = dataset_dir, transform = transform_test, train = False)

        n = len(dataset_train)
        i_train_val = [[] for _ in range(0, 100)]
        for i, (_, label) in enumerate(dataset_train):
            i_train_val[label].append(i)
        n_each_class = [int((n / 100) * (args.vaild_ratio + (1 - args.vaild_ratio) * (1 / args.imbalance) ** (t / 99))) for t in range(0, 100)]
        i_train_val = [random.sample(indexes, n_each_class[i]) for i, indexes in enumerate(i_train_val)]
        i_val = [random.sample(indexes, int(args.vaild_ratio * n / 100)) for i, indexes in enumerate(i_train_val)]
        
        i_train_val, i_val = sum(i_train_val, []), sum(i_val, [])
        dataset_vaild = Subset(dataset_train, i_val)
        dataset_train = Subset(dataset_train, complementary_list(i_train_val, i_val))


    return dataset_train, dataset_vaild, dataset_test

