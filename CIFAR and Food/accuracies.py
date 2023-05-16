import torch
import torch.nn.functional as F
import numpy as np

def get_acc(args):
    acc = None
    if args.accuracy == 'common_top_1':
        acc = common_top_1
    elif args.accuracy == 'test_time_aug_top_1':
        acc = test_time_aug_top_1
    return acc

def common_top_1(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.

    with torch.no_grad():
        p['model'].eval()
        for _, (img, labels) in enumerate(p[dataset_name + '_loader']):
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN':
                labels_one_hot = F.one_hot(labels, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100':
                labels_one_hot = F.one_hot(labels, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                labels_one_hot = F.one_hot(labels, 101).type(torch.float32)

            if torch.cuda.is_available():
                img, labels, labels_one_hot = img.cuda(), labels.cuda(), labels_one_hot.cuda()
            
            output = p['model'](args, img)
                
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item()
    
    accuracy = acc_number / len(p['dataset_' + dataset_name])
    loss /= len(p[dataset_name + '_loader'])
    return accuracy, loss

def test_time_aug_top_1(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.

    p['model'].eval()
    with torch.no_grad():
        for _, (img, labels) in enumerate(p[dataset_name + '_loader']):
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN':
                labels_one_hot = F.one_hot(labels, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100':
                labels_one_hot = F.one_hot(labels, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                labels_one_hot = F.one_hot(labels, 101).type(torch.float32)

            if torch.cuda.is_available():
                img, labels, labels_one_hot = img.cuda(), labels.cuda(), labels_one_hot.cuda()
            output = 0.5 * p['model'](args, img) + 0.5 * p['model'](args, p['aug'](img))
                
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item()
    
    accuracy = acc_number / len(p['dataset_' + dataset_name])
    loss /= len(p[dataset_name + '_loader'])
    return accuracy, loss