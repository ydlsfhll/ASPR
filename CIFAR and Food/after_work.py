from accuracies import get_acc

def after_train(args, p:dict, epoch:int, metric_logger):
    acc = get_acc(args)

    """ result = {'epoch': [],
            'train': {'acc': [], 'loss': []},
            'valid': {'acc': [], 'loss': []}, 
            'test': {'acc': [], 'loss': []}} """
    
    train_acc, train_loss = acc(args, p, 'train')
    valid_acc, valid_loss = acc(args, p, 'valid')
    test_acc, test_loss = acc(args, p, 'test')

    """ result['train']['acc'].append(train_acc)
    result['train']['loss'].append(train_loss)
    result['valid']['acc'].append(valid_acc)
    result['valid']['loss'].append(valid_loss)
    result['test']['acc'].append(test_acc)
    result['test']['loss'].append(test_loss) """
    
    metric_logger['epoch'].append(epoch)
    metric_logger['train']['loss'].append(train_loss)
    metric_logger['test']['loss'].append(test_loss)
    metric_logger['val']['loss'].append(valid_loss)
    metric_logger['train']['acc'].append(train_acc)
    metric_logger['test']['acc'].append(test_acc)
    metric_logger['val']['acc'].append(valid_acc)

    print()
    print('epoch:', epoch)
    print('train_acc:', train_acc, 'train_loss:', train_loss)
    print('valid_acc:', valid_acc, 'valid_loss:', valid_loss)
    print('test_acc:', test_acc, 'test_loss:', test_loss)
    print('learnging_rate:', p['lr_schedule'][p['lr_step'] - 1].item())
    
    return metric_logger