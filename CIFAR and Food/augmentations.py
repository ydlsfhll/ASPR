import torchvision.transforms as T
from torch.nn import ModuleList


def get_augmentation(args):
    
    if args.model_backbone == 'resnet18' or args.model_backbone == 'resnet50':

        augmentation = [T.RandomHorizontalFlip(),
                        T.RandomCrop(224, 28),
                        T.RandomRotation(15)]
        
    elif args.model_backbone == 'WideResNet-28-10' or args.model_backbone == 'WideResNet-40-2':
        
        augmentation = [T.RandomHorizontalFlip(),
                        T.RandomCrop(32, 4),
                        T.RandomRotation(15)]
    
    return augmentation