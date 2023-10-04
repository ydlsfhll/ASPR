from torch import nn
from torchvision import models
import torch.nn.functional as F


def get_num_classes(args):
    if args.dataset == 'PACS':
        return 7

def get_model(args):
    num_classes = get_num_classes(args)
    
    model = None
    
    if args.model_backbone == "resnet18":
        if args.pretrained_weights:
            model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet18(num_classes = num_classes, weights = None)
            
    elif args.model_backbone == 'resnet50':
        if args.pretrained_weights:
            model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet50(num_classes = num_classes, weights = None)
            
    return model