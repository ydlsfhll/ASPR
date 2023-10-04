from datetime import datetime
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', type = str, default = 'CIFAR10', 
                        help = 'dataset of the task')
    parser.add_argument('-o', '--optimizer', type = str, default = 'SGD', 
                        help = 'optimizer for the training')
    parser.add_argument('-m', '--model-backbone', type = str, default = 'resnet18', 
                        help = 'the backbone of the net')
    parser.add_argument('-mm', '--modified-model', action = 'store_true', default = False, 
                        help = 'true for the case that the model is modified (not original)')
    parser.add_argument('-w', '--pretrained-weights', action = 'store_true', default = False, 
                        help = 'use provided pretrained weights (if exists)')
    parser.add_argument('-e', '--train-epoches', type = int, default = 200, 
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
    parser.add_argument('-we', '--warm-up-epoches', type = int, default = 1, 
                        help = 'warm-up epochs of training phase')
    parser.add_argument('-a', '--accuracy', type = str, default = 'common_top_1', 
                        help = 'method for computing accuary')
    parser.add_argument('-v', '--vaild-ratio', type = float, default = 0.1, 
                        help = 'the ratio of vaild set divided in training set')
    parser.add_argument('-b', '--beta', type = float, default = 1, 
                        help = 'value of the coeffient beta in our loss')
    parser.add_argument('-s', '--save_model', action = 'store_true', default = False, 
                        help = 'save all the parameters of the model')
    parser.add_argument('-ct', '--continue_train', action = 'store_true', default = False, 
                        help = 'resume previous interrupted training')
    parser.add_argument('-cr', '--checkpoint_root', type = str, default = '', 
                        help = 'the file (path + filename) of previous interrupted training (needs continue_train == True)')
    parser.add_argument('-nt', '--nesterov', action = 'store_true', default = False, 
                        help = 'whether use nesterov')
    parser.add_argument('-i', '--imbalance', type = int, default = 10,
                        help = 'the imbalance of the long-tail CIFAR10/100')

    args = parser.parse_args()
    
    
    args.record_folder = '_'.join([str(datetime.now().date()),
                            args.dataset,
                            'supervise_test',
                            'baseline' if args.baseline else ''])
    args.record_name = '_'.join([str(datetime.now().hour),
                                str(datetime.now().minute),
                                str(datetime.now().second),
                                args.model_backbone,
                                args.loss_strategy,
                                args.optimizer,
                                'lr' + str(args.learning_rate),
                                'bs' + str(args.batch_size),
                                 'Warm' + str(args.warm_up_epoches),
                                'beta' + str(args.beta),
                                 'lrs-' + str(args.learning_rate_strategy),
                                             'i-' + str(args.imbalance)])
    
    return args