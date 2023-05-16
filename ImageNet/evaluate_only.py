from configurations import get_args
from preparations import fix_seed, get_dataloader, get_lr_schedule, adjust_lr
from datasets import get_dataset
from models import get_model
from criterion import get_criterion
from augmentations import get_augmentation
from accuracies import get_acc
from after_work import set_event_accumulators
from record import record_acc_and_loss

import sys, os, shutil, time
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.nn import Softmax
from torch.cuda import amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter


def main():
    args = get_args()
    p = pre_work(args)

    print('Start evaluating...')
    if args.resume:
        start_epoch = int(input('input start epoch:'))
        p['lr_step'] += start_epoch * len(p['train_loader'])
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.train_epochs)):
        load(args, p, epoch)
        evaluate(args, p, epoch)

def pre_work(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    fix_seed(2023)

    writer = get_summary_writer(args)

    dataset_train, dataset_valid, dataset_test = get_dataset(args)

    model = get_model(args)
    if torch.cuda.is_available():
        model.cuda()

    train_loader = get_dataloader(dataset_train, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)
    valid_loader = get_dataloader(dataset_valid, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)
    test_loader = get_dataloader(dataset_test, args.batch_size, True, args.num_workers, True, True, args.prefetch_factor)

    L = get_criterion(args)

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
    p['model'] = torch.compile(model)
    p['L'] = L
    p['lr_schedule'] = lr_schedule
    p['lr_step'] = 0
    p['aug'] = aug
    p['scaler'] = GradScaler()
    p['ea'] = None
    return p

def get_summary_writer(args):
    log_dir = input('input log folder (relative):')
    writer = SummaryWriter(log_dir = log_dir, filename_suffix = args.log_name_suffix)
    return writer

def load(args, p:dict, epoch:int):
    load_dir = p['writer'].get_logdir()
    names = os.listdir(load_dir)
    file_name = 'epoch_{0}.pt'.format(epoch)
    
    while file_name not in names:
        print('waiting for .pt file ...')
        time.sleep(10)
        names = os.listdir(load_dir)
    file_size = os.path.getsize(load_dir + r'/' + file_name)
    time.sleep(5)
    while (file_size_now := os.path.getsize(load_dir + r'/' + file_name)) > file_size:
        print('waiting for .pt file completely saved ...')
        file_size = file_size_now
        time.sleep(10)
    time.sleep(5) 
    checkpoint = torch.load(load_dir + r'/' + 'epoch_{0}.pt'.format(epoch))
    p['model'].load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        p['model'].cuda()
        
def evaluate(args, p: dict, epoch: int):
    print()
    print(r'evaluating ... epoch {0}/{1}'.format(epoch, args.train_epochs))
    acc = get_acc(args)

    p['lr_step'] += len(p['train_loader'])
    
    train_acc, train_loss = acc(args, p, 'train')
    valid_acc, valid_loss = acc(args, p, 'valid')
    test_acc, test_loss = acc(args, p, 'test')

    record_acc_and_loss([train_acc, valid_acc, test_acc], 
                        [train_loss, valid_loss, test_loss],
                        args, p, epoch)

    if p['ea'] == None:
        set_event_accumulators(p)
    
    source_file = p['writer'].get_logdir() + r'/epoch_{0}.pt'.format(epoch)
    if args.save_model:
        [ea.Reload() for ea in p['ea']]
        previous_valid_accs = [se.value for ea in p['ea'] for se in ea.Scalars('top_1_acc')][:-1]
        if len(previous_valid_accs) == 0:
            previous_best_valid_acc = 0.
        else:
            previous_best_valid_acc = max(previous_valid_accs) 
        
        print(previous_best_valid_acc, valid_acc[0])

        if previous_best_valid_acc <= valid_acc[0]:
            save_dir = p['writer'].get_logdir() + r'/best'
            shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok = True)
            save_name = r'_'.join(['epoch', str(epoch), 'valid_acc', str(valid_acc[0]), 'test_acc', str(test_acc[0])])
            shutil.move(source_file, save_dir)
            os.rename(save_dir + r'/epoch_{0}.pt'.format(epoch), save_dir + r'/{0}.pt'.format(save_name))
        else:
            os.remove(source_file)
    else:
        os.remove(source_file)
    

if __name__ == '__main__':
    main()
