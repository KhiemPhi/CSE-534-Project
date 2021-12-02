from .encoder import Encoder
from .segm import FCNHead

import torch.nn as nn 
import torch
from torch.utils.data import DataLoader


import pytorch_lightning as pl

class SaliencyModel(pl.LightningModule):

    def __init__(self, backbone_name, train_set, test_set, val_set, batch_size):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.batch_size = batch_size
    
   
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=50, pin_memory=True)    
    def val_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=50, pin_memory=True)    
    def test_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=50, pin_memory=True)    

    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  [p for p in self.model.parameters() if p.requires_grad]
        opt =  torch.optim.AdamW(params, lr=self.learning_rate)        #torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        
        return [opt], [sch] 
        
    def forward(self, img_batch):
        features = self.backbone(img_batch) 

        #1. Subnet #1: Segmentation subnet with task specific encoder


        #2. Subnet #2: Task Free Saliency Prediction - Report Saliency Metrics + Segm Metrics
        


        return features
    
    def training_step(self, batch, batch_idx):
        filename, website_img, seg_map, saliency_map, website_type = batch
        features = self.forward(website_img)
        return features
    
    
    


   
        
        
    

    