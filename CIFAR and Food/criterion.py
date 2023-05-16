from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim = 1):
        super().__init__()
        self.CosineSimilarity = nn.CosineSimilarity(dim, 1e-8)
    
    def forward(self, x, y):
        return -self.CosineSimilarity(x, y).mean()

def get_criterion(args):
    if args.loss_strategy == "MSELoss":
        loss = nn.MSELoss(reduction = 'mean')
        
    elif args.loss_strategy == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss(reduction = 'mean')
        
    elif args.loss_strategy == "NLL":
        loss = nn.NLLLoss(reduction = 'mean')
        
    elif args.loss_strategy == "CosineSimilarity":
        loss = CosineSimilarityLoss()
        
    elif args.loss_strategy == "L1Loss":
        loss = nn.L1Loss(reduction = 'mean')
        
    elif args.loss_strategy == "SmoothL1":
        loss = nn.SmoothL1Loss()
        
    elif args.loss_strategy == "L1Loss_and_CrossEntropyLoss":
        loss = [nn.L1Loss(reduction = 'mean'), nn.CrossEntropyLoss(reduction = 'mean')]
    
    elif args.loss_strategy == "MSELoss_and_CrossEntropyLoss":
        loss = [nn.MSELoss(reduction = 'mean'), nn.CrossEntropyLoss(reduction = 'mean')]
    
    return loss