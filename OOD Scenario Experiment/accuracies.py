import torch
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm

def get_acc(args):
    acc = None
    if args.accuracy == 'common_top_1':
        acc = common_top_1
    return acc

def common_top_1(args, p:dict, loader):
    data_number = 0
    acc_number = 0
    loss = 0.

    if args.dataset == 'PACS':
        normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    with torch.no_grad():
        p['model'].eval()
        for _, stacked_data in tqdm(enumerate(loader)):
            try:
                images = torch.cat([data[0] for data in stacked_data])
                labels = torch.cat([data[1] for data in stacked_data])
            except:
                images, labels = stacked_data
            data_number += images.shape[0]
            
            if args.dataset == 'PACS':
                images = normalize(images)
                labels_one_hot = F.one_hot(labels, 7).type(torch.float32)

            if torch.cuda.is_available():
                images, labels, labels_one_hot = images.cuda(), labels.cuda(), labels_one_hot.cuda()
            
            output = p['model'](images)
                
            acc_number += output.argmax(dim = 1).eq(labels).sum().item()
            loss += p['L'](output, labels_one_hot).item() * images.shape[0]
    
    accuracy = acc_number / data_number
    loss /= data_number
    return accuracy, loss

