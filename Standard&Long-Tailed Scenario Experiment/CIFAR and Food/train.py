from configurations import get_args
from preparations import pre_work, adjust_lr, picture, seed_torch, model_save
from after_work import after_train

from tqdm import tqdm

import sys, os
import random

import torch
from torch.nn import functional as F
import torchvision.transforms as T
from torch.nn import ModuleList, Softmax


def main():
    seed_torch(2023)
    args = get_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = r'{0}/{1}'.format(current_dir, args.record_folder)
    if not os.path.exists(log_dir):
        os.mkdir(r'{0}/{1}'.format(current_dir, args.record_folder))
    f = open(r'{0}/{1}/{2}.txt'.format(current_dir, args.record_folder, args.record_name), 'x')
    
    sys.stdout = f

    p = pre_work(args)

    print('Start training...')
    
    metric_logger = {'epoch':[],
                     'train':{'loss':[], 'acc':[], 'auc':[], 'ap':[], 'f1':[]},
                      'test':{'loss':[], 'acc':[], 'auc':[], 'ap':[], 'f1':[]},
                      'val':{'loss':[], 'acc':[], 'auc':[], 'ap':[], 'f1':[]}}
    best_val_acc = 0
    
    if args.continue_train:
        checkpoint = torch.load(args.checkpoint_root)
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = -1
    
    for epoch in tqdm(range(start_epoch+1, args.train_epoches), file = f):
        root_model = r'{0}/{1}/{2}'.format(current_dir, args.record_folder, args.record_name)
        train(args, p, epoch)
        result = after_train(args, p, epoch, metric_logger)
        root_pic = r'{0}/{1}/{2}.png'.format(current_dir, args.record_folder, args.record_name)
        picture(root_pic, result)
        
        if args.save_model == 1:
            if (result['val']['acc'][epoch-start_epoch-1] - best_val_acc >= 0) or (((epoch+1) % 50)==0):
                best_val_acc = result['val']['acc'][epoch-start_epoch-1]
                best_epoch = epoch
                model_save(root_model, p['model'], p['optimizer'], epoch, result)
    
def train(args, p:dict, epoch:int):
    p['model'].train()
    
    if args.continue_train == 1:
        checkpoint = torch.load(args.checkpoint_root)
        p['model'].load_state_dict(checkpoint['model_state_dict'])
        p['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    if args.baseline:
        
        for batch_idx, (img_spvd, label) in enumerate(p['train_loader']):
            
            adjust_lr(p)

            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN' or args.dataset == 'lt-CIFAR10':
                label = F.one_hot(label, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100' or args.dataset == 'lt-CIFAR100':
                label = F.one_hot(label, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                label = F.one_hot(label, 101).type(torch.float32)

            if torch.cuda.is_available():
                img_spvd, label = img_spvd.cuda(), label.cuda()

            p['optimizer'].zero_grad()

            aug = T.Compose(ModuleList(p['aug']))

            spvd = p['model'](args, aug(img_spvd))

            loss = p['L'](spvd, label)

            loss.backward()
            p['optimizer'].step()

            p['lr_step'] += 1

    else:
        for _, (img_spvd, label) in enumerate(p['train_loader']):
            adjust_lr(p)
            
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN' or args.dataset == 'lt-CIFAR10':
                label = F.one_hot(label, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100' or args.dataset == 'lt-CIFAR100':
                label = F.one_hot(label, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                label = F.one_hot(label, 101).type(torch.float32)

            if torch.cuda.is_available():
                img_spvd, label = img_spvd.cuda(), label.cuda()
            
            p['optimizer'].zero_grad()

            aug = T.Compose(ModuleList(p['aug']))
            
            aug_spvd = p['model'](args, aug(img_spvd))
            
            spvd = p['model'](args, img_spvd)
   
            softmax = Softmax(dim = 1)
            loss1 = p['L'](spvd, label)
            loss2 = torch.mean(torch.sum(label * (torch.log(softmax(spvd)) - torch.log(softmax(aug_spvd))), dim = 1))
            loss = loss1 + args.beta * loss2
            loss.backward()
            
            p['optimizer'].step()
            
            p['lr_step'] += 1

if __name__ == '__main__':
    main()