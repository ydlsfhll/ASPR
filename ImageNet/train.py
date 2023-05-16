from configurations import get_args
from preparations import pre_work, adjust_lr
from after_work import after_train

from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.nn import Softmax
from torch.cuda import amp


def main():
    args = get_args()
    p = pre_work(args)

    print('Start training...')

    for epoch in tqdm(range(0, args.train_epochs)):
        train(args, p, epoch)
        after_train(args, p, epoch)

def train(args, p: dict, epoch: int):
    print()
    print(r'training ... epoch {0}/{1}'.format(epoch, args.train_epochs))
    p['model'].train()

    for _, (images, labels) in tqdm(enumerate(p['train_loader'])):
        adjust_lr(p)

        labels = one_hot_label(labels, args)
        if torch.cuda.is_available():
            images, labels = images.cuda(non_blocking = True), labels.cuda(non_blocking = True)

        p['optimizer'].zero_grad(set_to_none = True)
        
        with amp.autocast():
            loss = forward(images, labels, args, p)

        p['scaler'].scale(loss).backward()
        p['scaler'].step(p['optimizer'])
        p['scaler'].update()

        p['lr_step'] += 1

def one_hot_label(labels, args):
    if args.dataset == 'CIFAR10':
        return F.one_hot(labels, 10).type(torch.float32)
    elif args.dataset == 'CIFAR100':
        return F.one_hot(labels, 100).type(torch.float32)
    elif args.dataset == 'ImageNet':
        return F.one_hot(labels, 1000).type(torch.float32)
    
def forward(images, labels, args, p:dict):
    if args.baseline:
        output = p['model'](images)
        loss = p['L'](output, labels)

    else:
        output = p['model'](images)
        output_aug = p['model'](p['aug'](images))
        softmax = Softmax(dim = 1)

        l1 = p['L'](output, labels)
        l2 = torch.mean(torch.sum(labels * (torch.log(softmax(output)) - torch.log(softmax(output_aug))), dim = 1))
        loss = l1 + args.beta * l2
    return loss


if __name__ == '__main__':
    main()
