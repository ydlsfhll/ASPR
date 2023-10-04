from datasets import get_dataset
from models import get_model
from criterion import get_criterion
from optimizers import get_optimizer
from augmentations import get_augmentation

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import random
import os
import numpy as np
from itertools import cycle


def pre_work(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    datasets = get_dataset(args)
    
    aug = get_augmentation(args)
    
    model = get_model(args)

    if torch.cuda.is_available():
        model.cuda()
    print(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of the Model\'s Parameters: ', num_params)

    L = get_criterion(args)

    optimizer = get_optimizer(args, model.parameters())

    scheduler = CosineAnnealingLR(optimizer, args.train_epochs)

    p = {}
    p['datasets'] = datasets
    p['model'] = model
    # p['model'] = torch.compile(model, mode = 'max-autotune', fullgraph = True)
    p['aug'] = aug
    p['L'] = L
    p['optimizer'] = optimizer
    p['scheduler'] = scheduler
    return p

def seed_torch(seed = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_loader(datasets, args):
    if type(datasets) == list:
        dataset_lengths = torch.tensor([len(dataset) for dataset in datasets])
        is_longest = (dataset_lengths - torch.max(dataset_lengths)) >= 0
        dataloaders = []
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, args.batch_size, True, num_workers = 4)
            if is_longest[i]:
                dataloaders.append(dataloader)
            else:
                dataloaders.append(cycle(dataloader))

        return zip(*dataloaders)
    
    else:
        return DataLoader(datasets, args.batch_size, True, num_workers = 4)


def model_reset(args, p:dict):
    del p['model']
    model = get_model(args)
    if torch.cuda.is_available():
        model.cuda()
    p['optimizer'] = get_optimizer(args, model.parameters())
    p['scheduler'] = CosineAnnealingLR(p['optimizer'], args.train_epochs)
    p['model'] = model
    # p['model'] = torch.compile(model, mode = 'max-autotune', fullgraph = True)

    