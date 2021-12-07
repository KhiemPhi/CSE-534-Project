# Module to train the base model, let's set up the dataset first by reading the json + matlab files
import argparse

from numpy.core.numeric import full
from dataset import WebSaliencyDataset, CustomWebSaliencyDataset
from model import SaliencyModel
import json
import copy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

website_type_dict = {1:"email",2:"fileshare",3:"job",4:"product", 5:"shopping", 6:"social", 7:"general"}
website_type_dict_reverse = {value: key for key, value in website_type_dict.items()}

def read_cfgs(cfg):

    cfg_file = open(cfg)
    cfg_dict = json.load(cfg_file)

    return cfg_dict


def main(args):
    print("----------- Web-Saliency Model ECCV 2015 -------------")
    
    #1. Grab the dataset object based on json inputs
    print("Processing JSON-Files")
    cfg = read_cfgs(args.config)    
    test_set = CustomWebSaliencyDataset(folder="testing", task=website_type_dict_reverse["social"])
    train_set = WebSaliencyDataset(matlab_path=cfg["matlab_path"], json_path=cfg["annotations_path"], imgs_dir=cfg["imgs_dir"], saliency_dir=cfg["saliency_path"], mode='train')
   
    checkpoint_callback = ModelCheckpoint(
            monitor='mean-IOU',
            dirpath="./checkpoints",
            filename='saliency-{epoch:02d}-{mean-IOU:.2f}',
            save_top_k=3,
            mode='max',
            save_last=True
    )           
        #2. Create trainer
    model = SaliencyModel( backbone_name=cfg["backbone_name"], 
                            train_set=train_set,test_set=test_set, val_set=train_set, 
                            batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"] )
    trainer = Trainer(gpus=[args.gpu] if torch.cuda.is_available() else "cpu", 
                      max_epochs=cfg["epochs"], auto_select_gpus=True, benchmark=True,        
                      auto_lr_find=True, check_val_every_n_epoch=10, num_sanity_val_steps=0, default_root_dir="./checkpoints",callbacks=[checkpoint_callback])

    #3. Train the model   
    trainer.tune(model=model)
    trainer.fit(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cfg", "--config", type=str,
                        default='configs/base.json', help="Path to the config to train the model")   
    parser.add_argument("-g", "--gpu", type=int,
                        default='0', help="GPU to use")
    parser.add_argument("-b", "--batch", type=int,
                        default=5, help="batch size to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-3, help="batch size to use")
    parser.add_argument("-ep", "--epochs", type=int,
                        default=20, help="batch size to use")    
 
 
    args = parser.parse_args()
    main(args)
