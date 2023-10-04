from preparations import get_loader
from accuracies import get_acc

import matplotlib.pyplot as plt

def after_train(args, p:dict, epoch:int, d:int, metric_logger):
    acc = get_acc(args)
    
    print('evaluating ...')
    dataset_train = [dataset for i, dataset in enumerate(p['datasets']) if not i == d]
    train_loader = get_loader(dataset_train, args)
    test_loader = get_loader(p['datasets'][d], args)

    train_acc, train_loss = acc(args, p, train_loader)
    test_acc, test_loss = acc(args, p, test_loader)
    
    metric_logger['epoch'].append(epoch)
    metric_logger['train']['loss'].append(train_loss)
    metric_logger['test']['loss'].append(test_loss)
    metric_logger['train']['acc'].append(train_acc)
    metric_logger['test']['acc'].append(test_acc)
    
    print()
    print('epoch:', epoch)
    print('train_acc:', train_acc, 'train_loss:', train_loss)
    print('test_acc:', test_acc, 'test_loss:', test_loss)
    
    root_pic = r'{0}/{1}_{2}.png'.format(args.record_folder, args.record_name, str(d))
    picture(root_pic, metric_logger, args.algorithm)
    
    return metric_logger

def picture(root, result, algorithm_name:str):
    plt.figure()
    train_loss = plt.plot(result['epoch'], result['train']['acc'], color = 'red', linestyle = '-.')
    test_loss = plt.plot(result['epoch'], result['test']['acc'], color = 'blue', linestyle = '--')
    plt.title(algorithm_name + ' accuracy per epoch (train:red, test:blue)')
    plt.savefig(root)
    plt.close()  