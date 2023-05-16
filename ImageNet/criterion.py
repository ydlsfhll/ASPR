from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction = 'mean', dim = 1):
        super().__init__()
        self.CosineSimilarity = nn.CosineSimilarity(dim, 1e-8)
        self.reduction = reduction

    def forward(self, x, y):
        if self.reduction == 'mean':
            return -self.CosineSimilarity(x, y).mean()
        elif self.reduction == 'sum':
            return self.CosineSimilarity(x, y).sum()

def get_criterion(args):
    loss = None

    if args.loss_strategy == "MSELoss":
        loss = nn.MSELoss(reduction = 'mean')
        
    elif args.loss_strategy == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss(reduction = 'mean')
        
    elif args.loss_strategy == "NLL":
        loss = nn.NLLLoss(reduction = 'mean')
        
    elif args.loss_strategy == "CosineSimilarity":
        loss = CosineSimilarityLoss(reduction = 'mean')
        
    elif args.loss_strategy == "L1Loss":
        loss = nn.L1Loss(reduction = 'mean')
        
    elif args.loss_strategy == "SmoothL1":
        loss = nn.SmoothL1Loss(reduction = 'mean')
     
    return loss