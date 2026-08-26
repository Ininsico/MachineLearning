import torch.optim as optim

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, optimizer, T_max, eta_min=0):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    
    def step(self):
        self.scheduler.step()
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
