from accuracies import get_acc

def after_train(args, p:dict, epoch:int, metric_logger):
    acc = get_acc(args)
    
    if args.accuracy == 'auc_ap_acc':
        train_acc, train_loss, train_auc, train_ap, train_f1 = acc(args, p, 'train')
        valid_acc, valid_loss, valid_auc, valid_ap, valid_f1 = acc(args, p, 'valid')
        test_acc, test_loss, test_auc, test_ap, test_f1 = acc(args, p, 'test')
    else:
        train_acc, train_loss = acc(args, p, 'train')
        valid_acc, valid_loss = acc(args, p, 'valid')
        test_acc, test_loss = acc(args, p, 'test')
    
    metric_logger['epoch'].append(epoch)
    metric_logger['train']['loss'].append(train_loss)
    metric_logger['test']['loss'].append(test_loss)
    metric_logger['val']['loss'].append(valid_loss)
    metric_logger['train']['acc'].append(train_acc)
    metric_logger['test']['acc'].append(test_acc)
    metric_logger['val']['acc'].append(valid_acc)
    
    if args.accuracy == 'auc_ap_acc':
        metric_logger['train']['auc'].append(train_auc)
        metric_logger['test']['auc'].append(test_auc)
        metric_logger['val']['auc'].append(valid_auc)
        metric_logger['train']['ap'].append(train_ap)
        metric_logger['test']['ap'].append(test_ap)
        metric_logger['val']['ap'].append(valid_ap)
        metric_logger['train']['f1'].append(train_f1)
        metric_logger['test']['f1'].append(test_f1)
        metric_logger['val']['f1'].append(valid_f1)
  

    print()
    print('epoch:', epoch)
    if args.accuracy == 'auc_ap_acc':
        print('train_acc:', train_acc, 'train_loss:', train_loss, 'train_auc:', train_auc, 'train_ap:', train_ap, 'train_f1:', train_f1)
        print('valid_acc:', valid_acc, 'valid_loss:', valid_loss, 'valid_auc:', valid_auc, 'valid_ap:', valid_ap, 'valid_f1:', valid_f1)
        print('test_acc:', test_acc, 'test_loss:', test_loss, 'test_auc:', test_auc, 'test_ap:', test_ap, 'test_f1:', test_f1)
    else:
        print('train_acc:', train_acc, 'train_loss:', train_loss)
        print('valid_acc:', valid_acc, 'valid_loss:', valid_loss)
        print('test_acc:', test_acc, 'test_loss:', test_loss)
    print('learnging_rate:', p['lr_schedule'][p['lr_step'] - 1].item())
    
    return metric_logger