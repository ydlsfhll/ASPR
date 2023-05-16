import torch
from torch import nn
from torchvision import models
from wide_resnet import Wide_ResNet


""" class Modified_resnet18(nn.Module):
    def __init__(self, original_resnet18, num_classes):
        super().__init__()
        in_features = original_resnet18.fc.in_features
        self.backbone = nn.Sequential(*(list(original_resnet18.children())[:-1]))
        self.linear1 = nn.Linear(in_features, 1024)
        self.linear2 = nn.Sequential(nn.AlphaDropout(0.2),
                                    nn.ReLU(True),
                                    nn.Linear(1024, num_classes))
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, args, *x):
        if len(x) > 1:
            spvd, aug_spvd, unspvd, aug_unspvd = x
            spvd, aug_spvd = self.backbone(spvd), self.backbone(aug_spvd)
            spvd, aug_spvd = torch.flatten(spvd, 1), torch.flatten(aug_spvd, 1)
            spvd, aug_spvd = self.linear1(spvd), self.linear1(aug_spvd)

            if args.loss_strategy == 'CosineSimilarity' or args.loss_strategy == 'MSE':
                pred = self.linear2(spvd)
                pred = self.softmax(pred)

            unspvd, aug_unspvd = self.backbone(unspvd), self.backbone(aug_unspvd)
            unspvd, aug_unspvd = torch.flatten(unspvd, 1), torch.flatten(aug_unspvd, 1)
            unspvd, aug_unspvd = self.linear1(unspvd), self.linear1(aug_unspvd)
            
            return pred, spvd, aug_spvd, unspvd, aug_unspvd

        else:
            x = self.backbone(*x)
            x = torch.flatten(x, 1)

            if args.loss_strategy == 'CosineSimilarity' or args.loss_strategy == 'MSE':
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.softmax(x)
            return x """

""" class Modified_resnet18(nn.Module):
    def __init__(self, original_resnet18, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(*(list(original_resnet18.children())[:-1]))

        in_features = original_resnet18.fc.in_features
        self.linear = nn.Sequential(nn.Linear(in_features, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(True),
                                    nn.AlphaDropout(0.2),
                                    nn.Linear(512, num_classes))
        
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, args, epoch:int, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        if epoch >= 0 and epoch < 40:
            return x
        else:
            x = self.linear(x)
            if args.loss_strategy == 'CosineSimilarity' or args.loss_strategy == 'MSE':
                x = self.softmax(x)
            return x """

class Modified_resnet18_Part_A(nn.Module):
    def __init__(self, original_resnet18):
        super().__init__()
        self.backbone = nn.Sequential(*(list(original_resnet18.children())[:-1]))

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class Modified_resnet18_Part_B(nn.Module):
    def __init__(self, original_resnet18, num_classes):
        super().__init__()
        in_features = original_resnet18.fc.in_features
       
        self.linear = torch.nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.linear(x)
        return x

class Modified_resnet18(nn.Module):
    def __init__(self, original_resnet18, num_classes):
        super().__init__()
        self.part_A = Modified_resnet18_Part_A(original_resnet18)
        self.part_B = Modified_resnet18_Part_B(original_resnet18, num_classes)
                
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, args, x):
        x = self.part_A(x)
        x = self.part_B(x)
        if args.loss_strategy == 'CosineSimilarity' or args.loss_strategy == 'MSE':
            x = self.softmax(x)
        return x

class Modified_resnet50_Part_A(nn.Module):
    def __init__(self, original_resnet50):
        super().__init__()
        
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
    def __init__(self, original_resnet50, num_classes):
        super().__init__()
        self.part_A = Modified_resnet50_Part_A(original_resnet50)
        self.part_B = Modified_resnet50_Part_B(original_resnet50, num_classes)
                
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, args, x):
        x = self.part_A(x)
        x = self.part_B(x)
        if args.loss_strategy == 'CosineSimilarity' or args.loss_strategy == 'MSE':
            x = self.softmax(x)
        return x

def get_num_classes(args):
    if args.dataset == 'CIFAR10' or args.dataset == 'SVHN':
        return 10
    elif args.dataset == 'CIFAR100':
        return 100
    elif args.dataset == 'Food101':
        return 101

def get_model(args):
    num_classes = get_num_classes(args)
    
    model = None
    
    if args.model_backbone == "resnet18":
        if args.pretrained_weights:
            model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(num_classes = num_classes, weights = None)
            
        if args.modified_model:
            model = Modified_resnet18(model, num_classes)
            
    elif args.model_backbone == 'resnet50':
        if args.pretrained_weights:
            model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(num_classes = num_classes, weights = None)
            
        if args.modified_model:
            model = Modified_resnet50(model, num_classes)
        
    elif args.model_backbone == 'WideResNet-28-10':
        model = Wide_ResNet(28, 10, 0.0, num_classes)
        
    elif args.model_backbone == 'WideResNet-40-2':
        model = Wide_ResNet(40, 2, 0.0, num_classes)
            
    return model