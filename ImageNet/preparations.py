from record import get_summary_writer, record_args, record_graph
from datasets import get_dataset
from models import get_model
from criterion import get_criterion
from optimizers import get_optimizer
from augmentations import get_augmentation

import random
import sys, os
import numpy as np
from math import pi, cos, sqrt

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler


def pre_work(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    fix_seed(2023)
    
    writer = get_summary_writer(args)
    record_args(writer, args)

    dataset_train, dataset_valid, dataset_test = get_dataset(args)
    
    model = get_model(args)

    train_loader = get_dataloader(dataset_train, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)
    valid_loader = get_dataloader(dataset_valid, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)
    test_loader = get_dataloader(dataset_test, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)

    if torch.cuda.is_available():
        model.cuda()
    record_graph(writer, model, torch.Tensor(args.batch_size, 3, 224, 224))
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_text('Number of the Model\'s Parameters', str(num_params))

    L = get_criterion(args)
    optimizer = get_optimizer(args, model.parameters())
    lr_schedule = get_lr_schedule(args, train_loader)

    aug = get_augmentation(args)

    p = {}
    p['writer'] = writer
    p['train_loader'] = train_loader
    p['valid_loader'] = valid_loader
    p['test_loader'] = test_loader
    p['dataset_train'] = dataset_train
    p['dataset_valid'] = dataset_valid
    p['dataset_test'] = dataset_test
    # p['model'] = model
    # p['model'] = torch.compile(model)
    p['model'] = torch.compile(model, fullgraph = True)
    # p['model'] = torch.compile(model, fullgraph = True, mode = 'max-autotune')
    # p['model'] = torch.compile(model, mode = 'max-autotune')
    p['L'] = L
    p['optimizer'] = optimizer
    p['lr_schedule'] = lr_schedule
    p['lr_step'] = 0
    p['aug'] = aug
    p['scaler'] = GradScaler()
    p['ea'] = None
    return p

def fix_seed(seed = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataloader(dataset, batch_size:int = 1, shuffle:bool = True, num_workers:int = 0, 
                   pin_memory:bool = False, drop_last:bool = False, prefetch_factor:int = 0):
    if dataset is None:
        return None
    else:
        return DataLoader(dataset, batch_size, shuffle, num_workers = num_workers, 
                          pin_memory = pin_memory, drop_last = drop_last, prefetch_factor = prefetch_factor,
                          persistent_workers = True)

def get_lr_schedule(args, train_loader):
    max_step = args.train_epochs * len(train_loader)
    warm_up_steps = args.warm_up_epochs * len(train_loader)
    lr_schedule = torch.zeros(max_step)

    for i in range(0, max_step):
        if warm_up_steps and i < warm_up_steps:
            lr_schedule[i] = args.learning_rate * i / warm_up_steps
            
        else:
            if args.learning_rate_strategy == 'cosine':
                T = 0
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * 0.5 * (1. + cos((2 * T + 1) * pi * percent_done_after_warm_up))

            elif args.learning_rate_strategy == 'linear':
                percent_left_after_warm_up = (max_step - i) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * percent_left_after_warm_up

            elif args.learning_rate_strategy == 'polynomial':
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                p = percent_done_after_warm_up ** 2
                lr_schedule[i] = args.learning_rate * (1 - 2 * p + p ** 2)

            elif args.learning_rate_strategy == 'round_high':
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * sqrt(1 - percent_done_after_warm_up ** 2)
            
            elif args.learning_rate_strategy == 'round_low':
                percent_left_after_warm_up = (max_step - i) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * (1 - sqrt(1 - percent_left_after_warm_up ** 2))

            elif args.learning_rate_strategy == 'milestone':
                gamma = 10
                if args.train_epochs == 200:
                    milestone = [0.3, 0.6, 0.65, 0.9, 1]
                elif args.train_epochs == 300:
                    milestone = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
                else:
                    milestone = [0.25, 0.5, 0.75, 1]
                milestone = [int(m * args.train_epochs * len(train_loader)) for m in milestone]
                for j, m in enumerate(milestone):
                    if i < m:
                        lr_schedule[i] = args.learning_rate / (gamma ** j)
                        break
            
            elif args.learning_rate_strategy == 'exponential':
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * (0.008 ** percent_done_after_warm_up)

            else:
                lr_schedule[i] = args.learning_rate

    return lr_schedule

def adjust_lr(p:dict):
    for param_group in p['optimizer'].param_groups:
        param_group['lr'] = p['lr_schedule'][p['lr_step']]
    p['writer'].add_scalar('lr', p['lr_schedule'][p['lr_step'] - 1].item(), p['lr_step'] - 1)