from .encoder import Encoder

import torch.nn as nn 
import torch


class TaskSpecificSaliencyModel(nn.Module):

    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        
    

    def forward(self, img_batch):
        features = self.backbone(img_batch)
        return features