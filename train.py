# Module to train the base model, let's set up the dataset first by reading the json + matlab files
import argparse
from dataset import WebSaliencyDataset
import json

def read_cfgs(cfg):

    cfg_file = open(cfg)
    cfg_dict = json.load(cfg_file)

    return cfg_dict


def main(args):
    print("----------- Web-Saliency Model ECCCV 2015 -------------")
    #1. Grab the dataset object based on json inputs
    cfg = read_cfgs(args.config)    
    full_dataset = WebSaliencyDataset(matlab_path=cfg["matlab_path"], json_path=cfg["annotations_path"], imgs_dir=cfg["imgs_dir"], saliency_dir=cfg["saliency_path"])




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
