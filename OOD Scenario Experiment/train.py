from configurations import get_args
from preparations import pre_work, seed_torch, get_loader, model_reset
from after_work import after_train

from tqdm import tqdm

import json, os, random

import torch
from torch.nn import functional as F
import torchvision.transforms as T
from torch.nn import ModuleList, Softmax


def main():
    args = get_args()
    seed_torch(args.seed)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)

    p = pre_work(args)

    print('Start training...')
    
    for d in range(0, len(p['datasets'])):
        metric_logger = {'epoch':[],
            'train':{'loss':[], 'acc':[]},
            'test':{'loss':[], 'acc':[]}}
        f = open(r'{0}/{1}_{2}.txt'.format(args.record_folder, args.record_name, str(d)), 'x')
        
        model_reset(args, p)
        for epoch in tqdm(range(0, args.train_epochs)):
            train(args, p, d)
            after_train(args, p, epoch, d, metric_logger)
        
        metric_logger['best acc'] = max(metric_logger['test']['acc'])
        json.dump(metric_logger, f)
        
def train(args, p:dict, d:int):
    p['model'].train()
    
    datasets = [dataset for i, dataset in enumerate(p['datasets']) if not i == d]
    if args.algorithm == 'ERM':
        train_loader = get_loader(datasets, args)
        for _, stacked_data in tqdm(enumerate(train_loader)):
            images = torch.cat([data[0] for data in stacked_data])
            labels = torch.cat([data[1] for data in stacked_data])
            images = p['aug'](images)
            if args.dataset == 'PACS':
                labels = F.one_hot(labels, 7).type(torch.float32)

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            p['optimizer'].zero_grad()
            output = p['model'](images)
            loss = p['L'](output, labels)
            loss.backward()
            p['optimizer'].step()
        p['scheduler'].step()
    
    elif args.algorithm == 'ours':
        train_loader = get_loader(datasets, args)
        softmax = Softmax(dim = 1)

        if args.dataset == 'PACS':
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        
        for _, stacked_data in tqdm(enumerate(train_loader)):
            images = torch.cat([data[0] for data in stacked_data])
            labels = torch.cat([data[1] for data in stacked_data])
            if args.dataset == 'PACS':
                images = normalize(images)
                labels = F.one_hot(labels, 7).type(torch.float32)
            images_aug = p['aug'](images)
            
            if torch.cuda.is_available():
                images, images_aug, labels = images.cuda(), images_aug.cuda(), labels.cuda()

            p['optimizer'].zero_grad()
            output, output_aug = p['model'](images), p['model'](images_aug)
            loss1 = p['L'](output, labels)
            loss2 = torch.mean(torch.sum(labels * (torch.log(softmax(output) + 1e-20) - torch.log(softmax(output_aug) + 1e-20)), dim = 1))
            loss = loss1 + args.beta * loss2
            loss.backward()
            p['optimizer'].step()
        p['scheduler'].step()
    

if __name__ == '__main__':
    main()