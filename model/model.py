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
        self.fcn_head = FCNHead(in_channels=2048, out_channels=5)
        self.task_specific_subnet = TaskSpecificSubnet()

        self.sal_loss_fn = nn.MSELoss(reduction='mean')
        self.seg_loss_fn = ClassBalancedFocalLoss(beta=0.9, gamma=0.05)
       
    
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=20, pin_memory=True, collate_fn=collate_fn)       
    def val_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=20, pin_memory=True, collate_fn=collate_fn)    
    def test_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=20, pin_memory=True, collate_fn=collate_fn)    

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

        #TODO: Visualize all these examples

      

        self.summarize_results()
        
        return features
    
    def on_validation_epoch_end(self):  
        self.summarize_results()
        self.log('mean-IOU', torch.tensor([self.mean_iou]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('global_acc', torch.tensor(self.acc_global), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    def on_test_start(self):
        self.setup_testing()
    
    def test_step(self,  batch: dict, _batch_idx: int):
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

        #TODO: Visualize all these examples

        self.summarize_results()
        
        return features
    
    def on_test_epoch_end(self):  
        self.summarize_results()
        self.log('mean-IOU', torch.tensor([self.mean_iou]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('global_acc', torch.tensor(self.acc_global), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    


   
        
        
    

    
