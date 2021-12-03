import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class ClassBalancedFocalLoss(nn.Module):

    def __init__(self, beta, gamma):
        super(ClassBalancedFocalLoss, self).__init__()
        self.gamma = gamma 
        self.beta = beta
        
    def forward(self, logits, target):

        
        samples_per_cls_base = torch.zeros(5).to(target.device).type(torch.float)
        samples_per_cls = torch.unique(target, return_counts=True)
        count = samples_per_cls[1].type(torch.float)
        samples_per_cls_base[samples_per_cls[0].type(torch.long)] = count       
        effective_num = 1.0 - torch.pow(samples_per_cls_base, torch.tensor(self.beta).type(torch.float).to(target.device))
        self.alpha = (1.0 - self.beta) / effective_num
        self.alpha = self.alpha / torch.sum(self.alpha) * 5
        
        loss = F.cross_entropy(logits, target, weight=self.alpha,
                               ignore_index=-100, reduction='mean',
                               )
       
       

        return loss
