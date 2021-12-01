import torch.nn as nn
import torch

import torchvision.transforms as T

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self):
        super().__init__()
        self.transforms = T.Compose([
            T.ToTensor()
        ])
        

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW       
        return x_out
