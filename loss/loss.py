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

        loss_func = nn.CrossEntropyLoss(weight=self.alpha, reduction='sum')
        loss = loss_func(logits, target)
        
        '''
        logits = logits.softmax(dim=1)
        labels_one_hot = F.one_hot(target, 5).float().reshape(logits.shape)    
       
        self.alpha = self.alpha.unsqueeze(0).repeat(logits.shape[0],1).unsqueeze(2).repeat(1,1,logits.shape[2]).unsqueeze(3).repeat(1,1,1,logits.shape[3])
        
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot,reduction = "none")
        modulator = torch.exp(-self.gamma * labels_one_hot * logits - self.gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))
        
        loss = modulator * BCLoss

        weighted_loss = self.alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels_one_hot)
        '''
       

        return loss
