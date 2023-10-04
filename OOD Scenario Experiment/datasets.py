import os
import random
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import torch


dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/dataset'

def get_dataset(args):
    if args.dataset == 'PACS':
        domain_folders = sorted(os.listdir(dataset_dir))
        return [ImageFolder(os.path.join(dataset_dir, folder), T.ToTensor()) for folder in domain_folders]

