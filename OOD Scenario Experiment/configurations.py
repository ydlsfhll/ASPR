from datetime import datetime
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', type = str, default = 'PACS', 
                        help = 'dataset of the task')
    parser.add_argument('-o', '--optimizer', type = str, default = 'AdamW', 
                        help = 'optimizer for the training')
    parser.add_argument('-m', '--model-backbone', type = str, default = 'resnet18', 
                        help = 'the backbone of the net')
    parser.add_argument('-mm', '--modified-model', action = 'store_true', default = False, 
                        help = 'true for the case that the model is modified (not original)')
    parser.add_argument('-pw', '--pretrained-weights', action = 'store_true', default = False, 
                        help = 'true for using Pytorch pretrained weights')
    parser.add_argument('-e', '--train-epochs', type = int, default = 50, 
                        help = 'total epochs of training phase')
    parser.add_argument('-al', '--algorithm', type = str, default = 'ERM', 
                        help = 'the alogorithm of the training')
    parser.add_argument('-l', '--loss-strategy', type = str, default = 'CrossEntropyLoss', 
                        help = '(main) loss function of the task')
    parser.add_argument('-lr', '--learning-rate', type = float, default = 5e-5, 
                        help = 'initial learning rate (after warm-up)')
    parser.add_argument('-bs', '--batch-size', type = int, default = 40, 
                        help = 'batch size for dataloaders')
    parser.add_argument('-a', '--accuracy', type = str, default = 'common_top_1', 
                        help = 'method for computing accuary')
    parser.add_argument('-b', '--beta', type = float, default = 0.5, 
                        help = 'value of the coeffient beta in our loss')
    parser.add_argument('-seed', '--seed', type = int, default = 2023, 
                        help = 'random seed')
    parser.add_argument('-lns', '--log-name-suffix', type = str, default = '', 
                        help = 'suffix of the name of the log folder')

    args = parser.parse_args()
    
    
    args.record_folder = '_'.join([str(datetime.now().date()),
                            args.dataset,
                            args.model_backbone,
                            args.algorithm,
                            args.log_name_suffix])
    args.record_name = '_'.join([str(datetime.now().hour),
                                str(datetime.now().minute),
                                str(datetime.now().second),
                                args.loss_strategy,
                                args.optimizer,
                                'lr' + str(args.learning_rate),
                                'bs' + str(args.batch_size),
                                'beta' + str(args.beta)])
    
    return args