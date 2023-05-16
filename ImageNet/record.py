import sys, os

import torch
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = r'/'.join([current_dir, args.log_folder_1, args.log_folder_2])
    os.makedirs(log_dir, exist_ok = True)
    writer = SummaryWriter(log_dir = log_dir, filename_suffix = args.log_name_suffix)

    return writer

def record_args(writer, args):
    for k, v in args.__dict__.items():
        writer.add_text(k, str(v))

def record_graph(writer, model, input_tensor = None):
    if input_tensor is None:
        input_tensor = torch.Tensor(1, 3, 32, 32)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    writer.add_graph(model, input_tensor)

def record_acc_and_loss(accs:list, losses:list, args, p:dict, epoch:int):
    if args.accuracy == 'common_top_1':
        acc_names = ['top_1']
    elif args.accuracy == 'common_top_5':
        acc_names = ['top_1', 'top_5']

    dataset_names = ['train', 'valid', 'test']

    for i, acc in enumerate(zip(*accs)):  
        dict_acc = dict(zip(dataset_names, acc))
        p['writer'].add_scalars('_'.join([acc_names[i], 'acc']), dict_acc, epoch)
    
    for i, acc in enumerate(accs):
        for j, a in enumerate(acc):
            print('_'.join([acc_names[j], dataset_names[i], 'acc:']), a, end = ' ')
        print('_'.join([dataset_names[i], 'loss:']), losses[i])

    p['writer'].add_scalars('loss', {'train': losses[0], 'valid': losses[1], 'test': losses[2]}, epoch)
    print('current_lr:', p['lr_schedule'][p['lr_step'] - 1].item())