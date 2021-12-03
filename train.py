# Module to train the base model, let's set up the dataset first by reading the json + matlab files
import argparse

from numpy.core.numeric import full
from dataset import WebSaliencyDataset
from model import SaliencyModel
import json
import copy
import torch
from pytorch_lightning import Trainer

def read_cfgs(cfg):

    cfg_file = open(cfg)
    cfg_dict = json.load(cfg_file)

    return cfg_dict


def main(args):
    print("----------- Web-Saliency Model ECCV 2015 -------------")
    
    #1. Grab the dataset object based on json inputs
    print("Processing JSON-Files")
    cfg = read_cfgs(args.config)    
    train_set = WebSaliencyDataset(matlab_path=cfg["matlab_path"], json_path=cfg["annotations_path"], imgs_dir=cfg["imgs_dir"], saliency_dir=cfg["saliency_path"], mode='train')
    test_set = WebSaliencyDataset(matlab_path=cfg["matlab_path"], json_path=cfg["annotations_path"], imgs_dir=cfg["imgs_dir"], saliency_dir=cfg["saliency_path"], mode='test')
    val_set = WebSaliencyDataset(matlab_path=cfg["matlab_path"], json_path=cfg["annotations_path"], imgs_dir=cfg["imgs_dir"], saliency_dir=cfg["saliency_path"], mode='val')
   
    #2. Create trainer
    model = SaliencyModel( backbone_name=cfg["backbone_name"], 
                            train_set=train_set,test_set=test_set, val_set=val_set, 
                            batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"]  )
    trainer = Trainer(gpus=[1] if torch.cuda.is_available() else "cpu", 
                      max_epochs=cfg["epochs"], auto_select_gpus=True, benchmark=True,        
                      auto_lr_find=True, check_val_every_n_epoch=10, num_sanity_val_steps=0)
    #3. Train the model
    trainer.fit(model=model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cfg", "--config", type=str,
                        default='configs/base.json', help="Path to the config to train the model")   
    parser.add_argument("-g", "--gpu", type=str,
                        default='0', help="GPU to use")
    parser.add_argument("-b", "--batch", type=int,
                        default=5, help="batch size to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-3, help="batch size to use")
    parser.add_argument("-ep", "--epochs", type=int,
                        default=20, help="batch size to use")    
 
 
    args = parser.parse_args()
    main(args)
