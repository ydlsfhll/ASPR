import torchvision.transforms as T
from torch.nn import ModuleList


def get_augmentation(args):
    if args.dataset == 'PACS':
        augmentation = [T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                        T.RandomGrayscale(),
                        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
    
    return T.Compose(augmentation)