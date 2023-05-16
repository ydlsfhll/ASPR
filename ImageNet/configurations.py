from datetime import datetime

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', type = str, default = 'CIFAR10', 
                        help = 'dataset of the task')
    parser.add_argument('-n', '--num-workers', type = int, default = 4, 
                        help = 'num_workers for dataloaders')
    parser.add_argument('-pf', '--prefetch-factor', type = int, default = 4, 
                        help = 'prefetch_factor for dataloaders')
    parser.add_argument('-o', '--optimizer', type = str, default = 'SGD', 
                        help = 'optimizer for the training')
    parser.add_argument('-m', '--model-backbone', type = str, default = 'resnet18', 
                        help = 'the backbone of the net')
    parser.add_argument('-mm', '--modified-model', action = 'store_true', default = False, 
                        help = 'true for the case that the model is modified (not original)')
    parser.add_argument('-w', '--pretrained-weights', action = 'store_true', default = False, 
                        help = 'use provided pretrained weights (if exists)')
    parser.add_argument('-e', '--train-epochs', type = int, default = 200, 
                        help = 'total epochs of training phase')
    parser.add_argument('-bl', '--baseline', action = 'store_true', default = False, 
                        help = 'true for conducting baseline training task')
    parser.add_argument('-l', '--loss-strategy', type = str, default = 'CrossEntropyLoss', 
                        help = '(main) loss function of the task')
    parser.add_argument('-lr', '--learning-rate', type = float, default = 0.1, 
                        help = 'initial learning rate (after warm-up)')
    parser.add_argument('-lrs', '--learning-rate-strategy', type = str, default = 'milestone', 
                        help = 'alteration strategy of the learning rate')
    parser.add_argument('-bs', '--batch-size', type = int, default = 100, 
                        help = 'batch size for dataloaders')
    parser.add_argument('-we', '--warm-up-epochs', type = int, default = 1, 
                        help = 'warm-up epochs of training phase')
    parser.add_argument('-a', '--accuracy', type = str, default = 'common_top_1', 
                        help = 'method for computing accuary')
    parser.add_argument('-v', '--vaild-ratio', type = float, default = 0.1, 
                        help = 'the ratio of vaild set divided in training set')
    parser.add_argument('-b', '--beta', type = float, default = 1, 
                        help = 'value of the coeffient beta in 我们的损失的名字叫啥')
    parser.add_argument('-s', '--save_model', action = 'store_true', default = False, 
                        help = 'save all the parameters of the model')
    parser.add_argument('-r', '--resume', action = 'store_true', default = False, 
                        help = 'resume previous interrupted training')
    parser.add_argument('-rf', '--resume-file', type = str, default = '', 
                        help = 'the file (path + filename) of previous interrupted training (needs resume == True)')
    parser.add_argument('-lns', '--log-name-suffix', type = str, default = '', 
                        help = 'the last part of the log name (as notes)')
    
    args = parser.parse_args()
    
    if args.resume:
        folders = args.resume_file.split('/')
        args.log_folder_1 = folders[-3]
        args.log_folder_2 = folders[-2]

    else:
        args.log_folder_1 = '_'.join([str(datetime.now().date()),
                                    args.dataset,
                                    'supervise_test',
                                    'baseline' if args.baseline else ''])
        args.log_folder_2 = '_'.join([str(datetime.now().hour),
                                    str(datetime.now().minute),
                                    str(datetime.now().second),
                                    args.model_backbone,
                                    args.loss_strategy,
                                    args.optimizer,
                                    'lr' + str(args.learning_rate),
                                    'bs' + str(args.batch_size),
                                    'beta' + str(args.beta) if not args.baseline else '',
                                    args.log_name_suffix])
    return args