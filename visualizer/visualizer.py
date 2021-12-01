import cv2
import numpy as np

import cv2
import numpy as np
from tqdm import tqdm

class Visualizer():

    def __init__(self, imgs, saliency_map_binaries, region_class_maps):
        self.imgs = imgs 
        self.saliency_maps = saliency_map_binaries
        self.region_class_maps = region_class_maps
    
    