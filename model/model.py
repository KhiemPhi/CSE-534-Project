from typing import MutableMapping
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .encoder import Encoder
from .segm import FCNHead, TaskSpecificSubnet
from .deeplab import DeepLabHead
from loss import ClassBalancedFocalLoss

from metrics import ConfusionMatrix, MetricLogger
from torchmetrics.functional import pearson_corrcoef

import torchvision 
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image 

def collate_fn(batch):
    """
    Gather all images in a batch

        Args:
            batch: a batch of dicts including img, target     
    """    
    return tuple(zip(*batch))

def stack_batches(item):
    return torch.stack(item)


class SaliencyModel(pl.LightningModule):

    def __init__(self, backbone_name, train_set, test_set, val_set, batch_size, learning_rate):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.batch_size = batch_size    
        self.learning_rate = learning_rate
        self.fcn_head = DeepLabHead(in_channels=2048, num_classes=5) #FCNHead(in_channels=2048, out_channels=5)
        self.task_specific_subnet = TaskSpecificSubnet()

        self.sal_loss_fn = nn.MSELoss(reduction='mean')
        self.seg_loss_fn =  ClassBalancedFocalLoss(beta=0.9, gamma=0.5)
        self.batch_num = 0
        self.CC = 0
       
    
    def setup_testing(self):
        self.confmat = ConfusionMatrix(num_classes=5)
        self.metric_logger = MetricLogger(delimiter="  ")

    
    def summarize_results(self):
            
        self.confmat.reduce_from_all_processes()
        acc_global, acc, iu = self.confmat.compute()
        self.acc_global = acc_global * 100
        self.mean_iou = iu.mean().item() * 100
   
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=5, pin_memory=True, collate_fn=collate_fn)       
    def val_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=5, pin_memory=True, collate_fn=collate_fn)    
    def test_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=5, pin_memory=True, collate_fn=collate_fn)    

    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params_backbone =  [p for p in self.backbone.parameters() if p.requires_grad]
        params_fcn = [p for p in self.fcn_head.parameters() if p.requires_grad]
        params_task_subnet = [p for p in self.task_specific_subnet.parameters() if p.requires_grad]
        params = params_backbone + params_fcn + params_task_subnet
        opt =  torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
        
        return [opt], [sch] 

    def forward(self, img_batch, seg_map, saliency_map, website_type):  
        input_shape = img_batch.shape[-2:]
        features = self.backbone(img_batch) 
        fcn_out = self.fcn_head(features)
      
        interp_out = F.interpolate(fcn_out, size=input_shape, mode="nearest")        

        #1. Subnet #1: Segmentation subnet with task specific encoder
        seg_map_pred = interp_out.argmax(1)
        task_specific_attn_shift  = self.task_specific_subnet(seg_map_pred.unsqueeze(1).to(torch.float), website_type.to(torch.float))

        #2. Subnet #2: Task Free Saliency Prediction 
        task_free_sal = interp_out.amax(1).unsqueeze(1).to(torch.float)
       
        #3. Combine two saliency maps to form final saliency map and do L2 Loss w/ respect to gt saliency map
        
        unified_atn_map = task_specific_attn_shift + task_free_sal  
        unified_atn_map = unified_atn_map.squeeze(1)

        #4. Perform Loss L2-Loss on Atn Map and Cross-Entropy on seg_map_pred  
      
        sal_loss = self.sal_loss_fn(unified_atn_map, saliency_map.type(torch.float)).to(torch.float)
        seg_loss = self.seg_loss_fn(interp_out.type(torch.float), seg_map.type(torch.long)).to(torch.float)
        
        losses = sal_loss + seg_loss
       
        return losses, sal_loss, seg_loss
    
    def training_step(self,  batch: dict, _batch_idx: int):
        filename, website_img, seg_map, saliency_map, website_type = batch
        website_img = torch.stack(website_img)
        seg_map = torch.stack(seg_map)
        saliency_map = torch.stack(saliency_map)
        website_type = torch.stack(website_type)
        
        multi_loss, sal_loss, seg_loss = self.forward(website_img, seg_map, saliency_map, website_type)
        self.log("seg-loss", seg_loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("sal-loss", sal_loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        
        return multi_loss


    def on_validation_start(self):
        self.setup_testing()
        self.batch_num = 0
    
    def validation_step(self,  batch: dict, _batch_idx: int):
        filename, website_img, seg_map, saliency_map, website_type = batch
        website_img = torch.stack(website_img)
        seg_map = torch.stack(seg_map)
        saliency_map = torch.stack(saliency_map)
        website_type = torch.stack(website_type)

        input_shape = website_img.shape[-2:]
        features = self.backbone(website_img) 
        fcn_out = self.fcn_head(features)
        interp_out = F.interpolate(fcn_out, size=input_shape, mode="nearest")

        #1. Subnet #1: Segmentation subnet with task specific encoder
        seg_map_pred = interp_out.argmax(1)
        self.confmat.update(seg_map.flatten(), seg_map_pred.flatten())

        task_specific_attn_shift  = self.task_specific_subnet(seg_map_pred.unsqueeze(1).to(torch.float), website_type.to(torch.float))

        #2. Subnet #2: Task Free Saliency Prediction 
        task_free_sal = interp_out.amax(1).unsqueeze(1).to(torch.float)
       
        #3. Combine two saliency maps to form final saliency map and do L2 Loss w/ respect to gt saliency map        
        unified_atn_map = task_specific_attn_shift + task_free_sal  
        unified_atn_map = unified_atn_map.squeeze(1)

        self.CC += pearson_corrcoef(unified_atn_map.flatten(), saliency_map.flatten())
        self.batch_num += 1

        self.summarize_results()
        
        return features
    
    def on_validation_epoch_end(self):  
        self.summarize_results()
        self.log('mean-IOU', self.mean_iou, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        self.log('mean-CC', self.CC/self.batch_num, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    def on_test_start(self):
        self.setup_testing()
    
    def test_step(self,  batch: dict, _batch_idx: int):

        if len(batch) == 3:
            
            filename, website_img,  website_type = batch
            website_img = torch.stack(website_img)
        
            website_type = torch.stack(website_type)

            input_shape = website_img.shape[-2:]
            features = self.backbone(website_img) 
            fcn_out = self.fcn_head(features)
            interp_out = F.interpolate(fcn_out, size=input_shape, mode="nearest")

            #1. Subnet #1: Segmentation subnet with task specific encoder
            seg_map_pred = interp_out.argmax(1)    
            task_specific_attn_shift  = self.task_specific_subnet(seg_map_pred.unsqueeze(1).to(torch.float), website_type.to(torch.float))

            #2. Subnet #2: Task Free Saliency Prediction 
            task_free_sal = interp_out.amax(1).unsqueeze(1).to(torch.float)
        
            #3. Combine two saliency maps to form final saliency map and do L2 Loss w/ respect to gt saliency map        
            unified_atn_map = task_specific_attn_shift + task_free_sal  
            unified_atn_map = unified_atn_map.squeeze(1)  

            test_map = None

            for img, atn_map, name in zip(website_img, unified_atn_map, filename):  
                atn_map = torch.abs(atn_map)
                atn_map /= torch.max(atn_map)
                atn_map[torch.where(atn_map < 0.6)] = 0
                grayscale = atn_map * 255
                grayscale_img = cv2.cvtColor(grayscale.cpu().numpy(), cv2.COLOR_GRAY2RGB)
                img = cv2.imread(name)
                img = cv2.resize(img, (1040,1040))
                stack = np.hstack((grayscale_img, img))
                cv2.imwrite(name[:-4]+"_combine.jpg", stack)
                
                fixation_map = torch.vstack(torch.where(grayscale!=0)).T
                values = atn_map[ fixation_map[:,0], fixation_map[:, 1]].unsqueeze(0).T           
                values = torch.abs(values)      
                values /= torch.max(values)

                fix_w_values = torch.hstack((fixation_map, values))
                fix_w_values = fix_w_values[fix_w_values[:, 0].sort(descending=False)[1]]
                fix_w_values = fix_w_values.cpu().numpy()
                fix_filename = name[:-4] + ".npy"
                np.save(fix_filename, fix_w_values)
                self.mean_iou = 0
                self.CC = 0

        else:
            filename, website_img, seg_map, saliency_map, website_type = batch

            website_img = torch.stack(website_img)
            seg_map = torch.stack(seg_map)
            saliency_map = torch.stack(saliency_map)
            website_type = torch.stack(website_type)

            input_shape = website_img.shape[-2:]
            features = self.backbone(website_img) 
            fcn_out = self.fcn_head(features)
            interp_out = F.interpolate(fcn_out, size=input_shape, mode="nearest")

            #1. Subnet #1: Segmentation subnet with task specific encoder
            seg_map_pred = interp_out.argmax(1)
            self.confmat.update(seg_map.flatten(), seg_map_pred.flatten())

            task_specific_attn_shift  = self.task_specific_subnet(seg_map_pred.unsqueeze(1).to(torch.float), website_type.to(torch.float))

            #2. Subnet #2: Task Free Saliency Prediction 
            task_free_sal = interp_out.amax(1).unsqueeze(1).to(torch.float)
        
            #3. Combine two saliency maps to form final saliency map and do L2 Loss w/ respect to gt saliency map        
            unified_atn_map = task_specific_attn_shift + task_free_sal  
            unified_atn_map = unified_atn_map.squeeze(1)
            unified_atn_map = torch.abs(unified_atn_map)
            unified_atn_map /= torch.max(unified_atn_map)
            unified_atn_map[torch.where(unified_atn_map < 0.3)] = 0
            unified_atn_map[torch.where(unified_atn_map !=0)] = 255

            
            

            self.CC += pearson_corrcoef(unified_atn_map.flatten(), saliency_map.flatten())
            self.batch_num += 1

            self.summarize_results()
                
        
       
    
    def on_test_epoch_end(self):  
        self.summarize_results()
        self.log('mean-IOU', self.mean_iou, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)        
        self.log('mean-CC', self.CC/self.batch_num, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        
    
    


   
        
        
    

    