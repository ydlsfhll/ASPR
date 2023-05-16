import torch
from torch import nn
from torchvision import models


class Modified_resnet18_Part_A(nn.Module):
    def __init__(self, original_resnet18):
        super().__init__()
        # original_resnet18.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        # del original_resnet18.maxpool
        
        self.backbone = nn.Sequential(*(list(original_resnet18.children())[:-1]))

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class Modified_resnet18_Part_B(nn.Module):
    def __init__(self, original_resnet18, num_classes):
        super().__init__()
        in_features = original_resnet18.fc.in_features

        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x

class Modified_resnet18(nn.Module):
    def __init__(self, original_resnet18, num_classes, args):
        super().__init__()
        self.part_A = Modified_resnet18_Part_A(original_resnet18)
        self.part_B = Modified_resnet18_Part_B(original_resnet18, num_classes)
                
        self.softmax = nn.Softmax(dim = 1)
        
        self.args = args

    def forward(self, x):
        x = self.part_A(x)
        x = self.part_B(x)
        if self.args.loss_strategy == 'CosineSimilarity' or self.args.loss_strategy == 'MSE':
            x = self.softmax(x)
        return x

class Modified_resnet50_Part_A(nn.Module):
    def __init__(self, original_resnet50):
        super().__init__()
        # original_resnet50.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        # del original_resnet50.maxpool
        
        self.backbone = nn.Sequential(*(list(original_resnet50.children())[:-1]))

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class Modified_resnet50_Part_B(nn.Module):
    def __init__(self, original_resnet50, num_classes):
        super().__init__()
        in_features = original_resnet50.fc.in_features
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x

class Modified_resnet50(nn.Module):
    def __init__(self, original_resnet50, num_classes, args):
        super().__init__()
        self.part_A = Modified_resnet50_Part_A(original_resnet50)
        self.part_B = Modified_resnet50_Part_B(original_resnet50, num_classes)
        
        self.softmax = nn.Softmax(dim = 1)
        
        self.args = args

    def forward(self, x):
        x = self.part_A(x)
        x = self.part_B(x)
        if self.args.loss_strategy == 'CosineSimilarity' or self.args.loss_strategy == 'MSE':
            x = self.softmax(x)
        return x

def get_num_classes(args):
    if args.dataset == 'CIFAR10':
        return 10
    elif args.dataset == 'CIFAR100':
        return 100
    elif args.dataset == 'ImageNet':
        return 1000

def get_model(args):
    num_classes = get_num_classes(args)
    
    model = None
    
    if args.model_backbone == "resnet18":
        if args.pretrained_weights:
            model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(num_classes = num_classes, weights = None)
            
        if args.modified_model:
            model = Modified_resnet18(model, num_classes, args)
            
    elif args.model_backbone == 'resnet50':
        if args.pretrained_weights:
            model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(num_classes = num_classes, weights = None)
            
        if args.modified_model:
            model = Modified_resnet50(model, num_classes, args)
            
    elif args.model_backbone == 'resnet101':
        if args.pretrained_weights:
            model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        else:
            model = models.resnet101(num_classes = num_classes, weights = None)

    return model