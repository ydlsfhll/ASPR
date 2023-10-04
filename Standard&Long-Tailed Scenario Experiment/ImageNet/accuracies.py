from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Softmax
from torch.cuda import amp


def get_acc(args):
    acc = None
    if args.accuracy == 'common_top_1':
        acc = common_top_1
    elif args.accuracy == 'common_top_5':
        acc = common_top_5
    return acc

def common_top_1(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.
    
    dataset = p['dataset_' + dataset_name]
    loader = p[dataset_name + '_loader']

    if dataset is None:
        return [0.], 0.

    with torch.no_grad():
        print(r'evaluating dataset_' + dataset_name)
        p['model'].eval()
        for _, (images, labels) in tqdm(enumerate(loader)):
            labels_one_hot = one_hot_label(labels, args)

            if torch.cuda.is_available():
                images, labels, labels_one_hot = images.cuda(non_blocking=True), labels.cuda(non_blocking=True), labels_one_hot.cuda(non_blocking=True)
            
            with amp.autocast():
                output, loss = forword(images, labels_one_hot, loss, args, p)
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
    
    accuracy = [acc_number / len(dataset)]
    loss /= len(loader)
    return accuracy, loss

def common_top_5(args, p:dict, dataset_name:str):
    acc_number_top_1 = 0
    acc_number_top_5 = 0
    loss = 0.
    
    dataset = p['dataset_' + dataset_name]
    loader = p[dataset_name + '_loader']

    if dataset is None:
        return [0., 0.], 0.

    with torch.no_grad():
        print(r'evaluating dataset_' + dataset_name + '...')
        p['model'].eval()
        for _, (images, labels) in tqdm(enumerate(loader)):
            labels_one_hot = one_hot_label(labels, args)

            if torch.cuda.is_available():
                images, labels, labels_one_hot = images.cuda(non_blocking=True), labels.cuda(non_blocking=True), labels_one_hot.cuda(non_blocking=True)
            
            with amp.autocast():
                output, loss = forword(images, labels_one_hot, loss, args, p)
            
            output = Softmax(dim = 1)(output)
            for k in range(0, 5):      
                idxmax = output.argmax(dim = 1)
                idxeq = idxmax.eq(labels)
                if k == 0:
                    acc_number_top_1 += idxeq.sum().item()
                acc_number_top_5 += idxeq.sum().item()

                output[torch.arange(0, output.shape[0], 1), idxmax] = -1
                idxeq = (idxeq == False)
                output, labels = output[idxeq], labels[idxeq]
    
    accuracy = [acc_number_top_1 / len(dataset), acc_number_top_5 / len(dataset)]
    loss /= len(loader)

    return accuracy, loss

def one_hot_label(labels, args):
    if args.dataset == 'CIFAR10':
        return F.one_hot(labels, 10).type(torch.float32)
    elif args.dataset == 'CIFAR100':
        return F.one_hot(labels, 100).type(torch.float32)
    elif args.dataset == 'ImageNet':
        return F.one_hot(labels, 1000).type(torch.float32)
    
def forword(images, labels_one_hot, loss, args, p:dict):
    if args.baseline:
        output = p['model'](images)
        loss += p['L'](output, labels_one_hot).item()
    else:
        output = p['model'](images)
        aug_output = p['model'](p['aug'](images))
        softmax = Softmax(dim = 1)
        l1 = p['L'](output, labels_one_hot)
        l2 = torch.mean(torch.sum(labels_one_hot * (torch.log(softmax(output)) - torch.log(softmax(aug_output))), dim = 1))
        loss += (l1 + args.beta * l2).item()
    return output, loss