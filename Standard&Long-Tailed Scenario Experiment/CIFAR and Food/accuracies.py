import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score

def get_acc(args):
    acc = None
    if args.accuracy == 'common_top_1':
        acc = common_top_1
    elif args.accuracy == 'test_time_aug_top_1':
        acc = test_time_aug_top_1
    elif args.accuracy == 'auc_ap_acc':
        acc = auc_ap_acc
    return acc

def common_top_1(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.

    with torch.no_grad():
        p['model'].eval()
        for _, (img, labels) in enumerate(p[dataset_name + '_loader']):
            
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN' or args.dataset == 'lt-CIFAR10':
                labels_one_hot = F.one_hot(labels, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100' or args.dataset == 'lt-CIFAR100':
                labels_one_hot = F.one_hot(labels, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                labels_one_hot = F.one_hot(labels, 101).type(torch.float32)

            if torch.cuda.is_available():
                img, labels, labels_one_hot = img.cuda(), labels.cuda(), labels_one_hot.cuda()
            
            output = p['model'](args, img)
                
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item()
    
    accuracy = acc_number / len(p['dataset_' + dataset_name])
    loss /= len(p[dataset_name + '_loader'])
    return accuracy, loss

def test_time_aug_top_1(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.

    p['model'].eval()
    with torch.no_grad():
        for _, (img, labels) in enumerate(p[dataset_name + '_loader']):
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN' or args.dataset == 'lt-CIFAR10':
                labels_one_hot = F.one_hot(labels, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100' or args.dataset == 'lt-CIFAR100':
                labels_one_hot = F.one_hot(labels, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                labels_one_hot = F.one_hot(labels, 101).type(torch.float32)

            if torch.cuda.is_available():
                img, labels, labels_one_hot = img.cuda(), labels.cuda(), labels_one_hot.cuda()
            output = 0.5 * p['model'](args, img) + 0.5 * p['model'](args, p['aug'](img))
                
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item()
    
    accuracy = acc_number / len(p['dataset_' + dataset_name])
    loss /= len(p[dataset_name + '_loader'])
    return accuracy, loss


def auc_ap_acc(args, p:dict, dataset_name:str):
    acc_number = 0
    loss = 0.
    probs_all, gt_all = None, np.array([])

    with torch.no_grad():
        p['model'].eval()
        for _, (img, labels) in enumerate(p[dataset_name + '_loader']):
   
            if args.dataset == 'CIFAR10' or args.dataset == 'SVHN' or args.dataset == 'lt-CIFAR10':
                labels_one_hot = F.one_hot(labels, 10).type(torch.float32)
            elif args.dataset == 'CIFAR100' or args.dataset == 'lt-CIFAR100':
                labels_one_hot = F.one_hot(labels, 100).type(torch.float32)
            elif args.dataset == 'Food101':
                labels_one_hot = F.one_hot(labels, 101).type(torch.float32)

            if torch.cuda.is_available():
                img, labels, labels_one_hot = img.cuda(), labels.cuda(), labels_one_hot.cuda()
            
            output = p['model'](args, img)

            gt_all = labels_one_hot.detach().cpu().numpy() if probs_all is None else np.concatenate((gt_all, labels_one_hot.detach().cpu().numpy()), axis=0)   # Logging Information
            
            probs_np = output.detach().cpu().numpy()
            
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)   # Logging Information
            
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item()
    
    pred_test = [probs_all, gt_all]
    indicator = GradMetrics(pred_test, avg='micro')
            
    auc_test = float(indicator[0]) 
    ap_test = float(indicator[1]) 
    f1_test = float(indicator[2]) 
    
    accuracy = acc_number / len(p['dataset_' + dataset_name])
    loss /= len(p[dataset_name + '_loader'])
    return accuracy, loss, auc_test, ap_test, f1_test


def GradMetrics(pred, avg='micro'):
    grade_pred, grade = np.array(pred[0]), np.array(pred[1])
    grade_oh = grade
    rocauc = roc_auc_score(grade_oh, grade_pred, average=avg)
    ap = average_precision_score(grade_oh, grade_pred, average=avg)
    f1 = 0
        
    return np.array([str("{0:.4f}".format(rocauc)), str("{0:.4f}".format(ap)), str("{0:.4f}".format(f1))])