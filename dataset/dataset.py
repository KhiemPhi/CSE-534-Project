from torch.utils.data import Dataset
import scipy.io
import numpy as np
import json
import os
import cv2
import imgviz
from visualizer import get_sal_map, generate_mask_from_bbox, get_sal_map_heat
from tqdm import tqdm
import torch

from visualizer.vis_utils import visualize_seg_map
from .data_augmentation import DataAugmentation, SegMapResizer
import torchvision.transforms as T


sepearate_dataset_key = "eccv"
website_type_dict = {1:"email",2:"fileshare",3:"job",4:"product", 5:"shopping", 6:"social", 7:"general"}
website_type_dict_reverse = {value: key for key, value in website_type_dict.items()}
website_element_dict = {"Background":0, "Button":1, "Text":2, "Input Field":3, "Image":4}
website_element_dict_reverse = {value: key for key, value in website_element_dict.items()}


class WebSaliencyDataset(Dataset):

    def __init__(self, matlab_path, json_path, imgs_dir, saliency_dir, vis=True, vis_dir="vis_gt", mode='train'):
        super().__init__()

        #1. Process the Matlab files to get the images + saliency regions
        self.eccv_data = scipy.io.loadmat(matlab_path)["webpages"]

        self.eccv_data =  [ x[0][0][0] for x in self.eccv_data]

        self.eccv_categories = list(np.unique([ x[3] for x in self.eccv_data]))

        #2. Parse the JSON file to get the correct annotations, we build a list of tuple that goes:
        # file_name, img, seg map, saliency map, website_type
        self.dataset = []     

        self.json_path = json_path
        self.imgs_dir = imgs_dir
        self.saliency_dir = saliency_dir
        self.vis = vis
        self.vis_dir = vis_dir
        self.seg_map_resizer = SegMapResizer()
        
        self.mode = mode

        self.process_eccv_data()
        self.process_annotations()   
        self.data_augmentation()

    def data_augmentation(self):       
        self.dataset = [  (x[0], self.transforms(x[1]), x[2], self.seg_map_resizer(x[3]), x[4]) for x in self.dataset]
       
    
    def form_seg_map(self, annotation, saliency_map):
      
        if len(annotation["regions"]) == 0: 
            seg_map = np.zeros_like(saliency_map) # empty seg map
        else: 
            regions = annotation["regions"]
            seg_map = np.zeros_like(saliency_map) # empty seg map
            seg_map = generate_mask_from_bbox(regions, seg_map, website_element_dict)               
        
        return seg_map
    
    def process_annotations(self):
        
        annotation_values = []
        for path in self.json_path:
            annotation_file = open(path)
            annotation_dict = json.load(annotation_file)
            annotation_values.extend(list(annotation_dict.values()))

        if self.mode == 'train':
            annotation_values = annotation_values[0: int( len(annotation_values) * 0.6 ) ]
        elif self.mode == 'test':
            annotation_values = annotation_values[int( len(annotation_values) * 0.6 ):int( len(annotation_values) * 0.8 )  ]
        elif self.mode == "val": 
            annotation_values = annotation_values[int( len(annotation_values) * 0.8 ): ]   
        
        mean = np.array([0,0,0]).astype(np.float64)
        std = np.array([0,0,0]).astype(np.float64)
    

        for annotation in tqdm(annotation_values):
            if sepearate_dataset_key in annotation["filename"]:
                # ---> processs eccv data differently
                idx = self.name_to_items_eecv[annotation["filename"]] 
                webpage = self.eccv_data[idx]

                website_img = webpage[0]
                website_img = cv2.cvtColor(website_img, cv2.COLOR_BGR2RGB)
                web_eye_gaze = webpage[1].squeeze(0)[0]
                category = webpage[3][0]           

                H, W, _ = website_img.shape    

                web_eye_gaze = web_eye_gaze -  web_eye_gaze.min()
                web_eye_gaze = web_eye_gaze / web_eye_gaze.max()
                web_eye_gaze[:,0] *= W
                web_eye_gaze[:,1] *= H

                saliency_map = get_sal_map_heat(web_eye_gaze, H, W)
                website_type = np.zeros(shape=len(website_type_dict.keys()))
                website_type[website_type_dict_reverse[category]] = 1
                seg_map = self.form_seg_map(annotation, saliency_map)
                seg_map = cv2.resize(seg_map, dsize=(1040,1040))  
                mean += website_img.mean(axis=(0,1))
                std +=  website_img.std(axis=(0,1))
            
            else: 
                # ----> souradeep dataset                
                website_type = np.array([0,0,0,0,0,0,1]) # one hot encoding for 7
                website_path = os.path.join(self.imgs_dir, annotation["filename"])
                saliency_map_path = os.path.join(self.saliency_dir, annotation["filename"])
                saliency_map = cv2.imread(saliency_map_path, flags=cv2.IMREAD_GRAYSCALE)
                website_img = cv2.imread(website_path)
                seg_map = self.form_seg_map(annotation, saliency_map)  
                seg_map = cv2.resize(seg_map, dsize=(1040,1040))             
                mean += website_img.mean(axis=(0,1))
                std +=  website_img.std(axis=(0,1))
               
            '''
            if self.vis:                       
                saliency_map_rgb = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2RGB)
                seg_map_visualization = visualize_seg_map(seg_map, website_img)
                seg_map_vis = np.hstack((website_img, seg_map_visualization, saliency_map_rgb))
                full_file_name = os.path.join(self.vis_dir, "vis_gt_{}".format(annotation["filename"]))
                cv2.imwrite(full_file_name, seg_map_vis)
            '''
            
            website_type = torch.from_numpy(website_type)
            seg_map = torch.from_numpy(seg_map)
            self.dataset.append( (annotation["filename"], website_img, seg_map, saliency_map, website_type)  )
            
            
        
        self.mean = mean / len(annotation_dict.values()) 
        self.std = std / len(annotation_dict.values()) 
      
        self.transforms = DataAugmentation(mean=self.mean, std=self.std)
         

    def process_eccv_data(self):

        category_count = {}
        
        for i in self.eccv_categories:
            category_count[str(i)] = 0
      
        
        self.name_to_items_eecv = {}
        for idx, webpage in enumerate(self.eccv_data):
            category = webpage[3]
            category_count[str(i)] += 1
            count = str(category_count[str(i)])
            basename = 'eccv_2015_{}_{}.jpg'.format(category[0], count)            
            self.name_to_items_eecv[basename] = idx

           
    def __len__(self):
        """
        Overwrite the len function to get the number of images in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx:int):
        return self.dataset[idx]

 

   
   

       
        

        
        

