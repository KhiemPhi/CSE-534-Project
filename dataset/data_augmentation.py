from torch.functional import _return_counts
import torch.nn as nn
import torch

import torchvision.transforms as T

import torchvision.transforms.functional as f

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((1040,1040)),
            T.Normalize(mean=self.mean, std=self.std),
                      
        ])
        

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW       
        return x_out

class SegMapResizer(nn.Module):

    def __init__(self):
        super().__init__()
       
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((1040,1040)),                         
        ])
        

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW   
        x_out = x_out.squeeze(0)
        return x_out
