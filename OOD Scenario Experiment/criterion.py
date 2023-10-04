from torch import nn


def get_criterion(args):
    if args.loss_strategy == "MSELoss":
        loss = nn.MSELoss(reduction = 'mean')
        
    elif args.loss_strategy == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss(reduction = 'mean')
        
    elif args.loss_strategy == "NLL":
        loss = nn.NLLLoss(reduction = 'mean')
        
    elif args.loss_strategy == "L1Loss":
        loss = nn.L1Loss(reduction = 'mean')
        
    elif args.loss_strategy == "SmoothL1":
        loss = nn.SmoothL1Loss(reduction = 'mean')
   
    return loss