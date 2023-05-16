import torchvision.transforms as T


def get_augmentation(args):
    augmentation = None
    if args.dataset in ['CIFAR10', 'CIFAR100', 'ImageNet']:

        augmentation = [T.RandomHorizontalFlip(),
                        T.RandomCrop(224, 28),
                        T.RandomRotation(15)]
        
        augmentation = T.Compose(augmentation)
    return augmentation