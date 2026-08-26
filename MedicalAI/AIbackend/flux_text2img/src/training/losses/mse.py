import torch.nn as nn

class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.loss(pred, target)
