from accuracies import get_acc
from record import record_acc_and_loss

import sys, os

import torch
from tensorboard.backend.event_processing import event_accumulator

def after_train(args, p:dict, epoch:int):
    acc = get_acc(args)

    train_acc, train_loss = acc(args, p, 'train')
    valid_acc, valid_loss = acc(args, p, 'valid')
    test_acc, test_loss = acc(args, p, 'test')

    record_acc_and_loss([train_acc, valid_acc, test_acc], 
                        [train_loss, valid_loss, test_loss],
                        args, p, epoch)
    
    if p['ea'] == None:
        set_event_accumulators(p)

    if args.save_model:
        [ea.Reload() for ea in p['ea']]
        previous_valid_accs = [se.value for ea in p['ea'] for se in ea.Scalars('top_1_acc')][:-1]
        if len(previous_valid_accs) == 0:
            previous_best_valid_acc = 0.
        else:
            previous_best_valid_acc = max(previous_valid_accs) 
        
        save_name = r'_'.join(['epoch', str(epoch), 'valid_acc', str(valid_acc[0]), 'test_acc', str(test_acc[0])]) + '.pt'
        if previous_best_valid_acc <= valid_acc[0]:
            save_dir = p['writer'].get_logdir() + r'/best'
            save_model(save_dir, save_name, p, epoch)
        
        elif (epoch % 50) == 0:
            save_dir = p['writer'].get_logdir()
            save_model(save_dir, save_name, p, epoch)

def set_event_accumulators(p:dict):
    valid_acc_dir = p['writer'].get_logdir() + r'/top_1_acc_valid'
    p['ea'] = []
    file_list = sorted(os.listdir(valid_acc_dir), key = lambda x: os.path.getctime(os.path.join(valid_acc_dir, x)))
    for name in file_list:
        if 'events.out.tfevents.' in name:
            ea = event_accumulator.EventAccumulator(r'{0}/{1}'.format(valid_acc_dir, name))
            ea.Reload()
            p['ea'].append(ea)
            
def save_model(save_dir:str, save_name:str, p:dict, epoch:int):
    os.makedirs(save_dir, exist_ok = True)
    for name in os.listdir(save_dir):
        if '.pt' in name:
            os.remove(r'/'.join([save_dir, name]))
    
    torch.save({'epoch': epoch, 
                'model_state_dict': p['model'].state_dict(),  
                'optimizer_state_dict': p['optimizer'].state_dict()}, r'/'.join([save_dir, save_name]))