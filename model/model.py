from .encoder import Encoder
from .segm import FCNHead

import torch.nn as nn 
import torch


class TaskSpecificSaliencyModel(nn.Module):

    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        
        
    

    def forward(self, img_batch):
        features = self.backbone(img_batch) 

        #1. Subnet #1: Segmentation subnet with task specific encoder


        #2. Subnet #2: Task Free Saliency Prediction - Report Saliency Metrics + Segm Metrics

        


        return features