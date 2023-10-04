from torch.utils.data import DataLoader
from datasets import get_dataset
from models import get_model
from criterion import get_criterion
from optimizers import get_optimizer
from augmentations import get_augmentation
from math import pi, cos, sqrt
import matplotlib.pyplot as plt
import torch
import random
import os
import numpy as np


def get_lr_schedule(args, train_loader):
    max_step = args.train_epoches * len(train_loader)
    warm_up_steps = args.warm_up_epoches * len(train_loader)
    lr_schedule = torch.zeros(max_step)

    for i in range(0, max_step):
        if warm_up_steps and i < warm_up_steps:
            lr_schedule[i] = args.learning_rate * i / warm_up_steps
            
        else:
            if args.learning_rate_strategy == 'cosine':
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * 0.5 * (1. + cos(pi * percent_done_after_warm_up))

            elif args.learning_rate_strategy == 'linear':
                percent_left_after_warm_up = (max_step - i) / (max_step - warm_up_steps)
                lr_schedule[i] = args.learning_rate * percent_left_after_warm_up

            elif args.learning_rate_strategy == 'milestone':
                milestone = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
                milestone = [int(m * args.train_epoches * len(train_loader)) for m in milestone]
                for j, m in enumerate(milestone):
                    if i < m:
                        lr_schedule[i] = args.learning_rate / (5 ** j)
                        break
            
            elif args.learning_rate_strategy == 'milestone2':
                milestone = [0.3, 0.6, 0.8, 1]
                milestone = [int(m * args.train_epoches * len(train_loader)) for m in milestone]
                for j, m in enumerate(milestone):
                    if i < m:
                        lr_schedule[i] = args.learning_rate / (5 ** j)
                        break
                        
            elif args.learning_rate_strategy == 'milestone3':
                milestone = [0.4, 0.6, 0.8, 1]
                milestone = [int(m * args.train_epoches * len(train_loader)) for m in milestone]
                for j, m in enumerate(milestone):
                    if i < m:
                        lr_schedule[i] = args.learning_rate / (10 ** j)
                        break 
            
            elif args.learning_rate_strategy == 'exponential':
                percent_done_after_warm_up = (i - warm_up_steps) / (max_step - warm_up_steps)
                lr_schedule[i]= args.learning_rate * (0.0016 ** percent_done_after_warm_up)

            else:
                lr_schedule[i] = args.learning_rate
                
    return lr_schedule

def pre_work(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(args)

    dataset_train, dataset_valid, dataset_test = get_dataset(args)
    
    model = get_model(args)

    train_loader = DataLoader(dataset_train, args.batch_size, True, num_workers = 8, pin_memory = True, drop_last = True)
    valid_loader = DataLoader(dataset_valid, args.batch_size, True, num_workers = 8, pin_memory = True, drop_last = True)
    test_loader = DataLoader(dataset_test, args.batch_size, True, num_workers = 8, pin_memory = True, drop_last = True)

    if torch.cuda.is_available():
        model.cuda()
    print(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of the Model\'s Parameters: ', num_params)

    L = get_criterion(args)
    optimizer = get_optimizer(args, model.parameters())
    lr_schedule = get_lr_schedule(args, train_loader)

    aug = get_augmentation(args)

    p = {}
    p['train_loader'] = train_loader
    p['valid_loader'] = valid_loader
    p['test_loader'] = test_loader
    p['dataset_train'] = dataset_train
    p['dataset_valid'] = dataset_valid
    p['dataset_test'] = dataset_test
    # p['model'] = model
    p['model'] = torch.compile(model, fullgraph = True)
    p['L'] = L
    p['optimizer'] = optimizer
    p['lr_schedule'] = lr_schedule
    p['lr_step'] = 0
    p['aug'] = aug
    return p

def adjust_lr(p:dict):
    for param_group in p['optimizer'].param_groups:
            param_group['lr'] = p['lr_schedule'][p['lr_step']]

def picture(root, result):
    plt.figure()
    train_loss = plt.plot(result['epoch'], result['train']['acc'], color='red', linestyle='-.')
    test_loss = plt.plot(result['epoch'], result['test']['acc'], color='blue', linestyle='--')
    if result['val']['acc'][-1] != 0:
        val_loss = plt.plot(result['epoch'], result['val']['acc'], color='green', linestyle='-')
    plt.title('acc vs. epoch(train:red, test:blue)')

    plt.savefig(root)
    
    
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def model_save(root, model, optimizer, epoch, result):
    os.makedirs(root, exist_ok=True)
    torch.save({
        'epoch': epoch,  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'result': result
    }, root + '/' + 'epoch' + str(epoch) + '_val_' + str(result['val']['acc'][epoch]) + '_test_' + str(result['test']['acc'][epoch]) + "_.pt")
    