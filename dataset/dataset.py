from torch.utils.data import Dataset
import scipy.io
import numpy as np
import json
import os
import cv2
import imgviz


sepearate_dataset_key = "eccv"
website_type_dict = {1:"Shopping",2:"Browsing",3:"Form Filling",4:"Signing up", 5:"General"}
website_element_dict = {"Background":0, "Button":1, "Text":2, "Input Field":3, "Image":4}

website_element_dict_reverse = {0:"Background", 1:"Button", 2:"Text", 3:"Input Field", 4:"Image"}


class WebSaliencyDataset(Dataset):

    def __init__(self, matlab_path, json_path, imgs_dir, saliency_dir, vis=True, vis_dir="vis_gt"):
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

        self.process_eccv_data()
        self.process_annotations()
    
    def process_annotations(self):

        annotation_file = open(self.json_path)
        annotation_dict = json.load(annotation_file)

        for annotation in annotation_dict.values():
            if sepearate_dataset_key in annotation["filename"]:
                # ---> processs eccv data differently
                pass 
            else: 
                # ----> souradeep dataset                
                website_type = np.array([0,0,0,0,1]) # one hot encoding for 5
                website_path = os.path.join(self.imgs_dir, annotation["filename"])
                saliency_map_path = os.path.join(self.saliency_dir, annotation["filename"])
                saliency_map = cv2.imread(saliency_map_path, flags=cv2.IMREAD_GRAYSCALE)
                website_img = cv2.imread(website_path)

                # Now build the seg-map
                if len(annotation["regions"]) == 0: 
                    seg_map = np.zeros_like(saliency_map) # empty seg map
                else: 
                    regions = annotation["regions"]
                    seg_map = np.zeros_like(saliency_map) # empty seg map
                    region_names = []
                    region_labels = []

                    for i in regions:
                        try:
                            mask_value = website_element_dict[i["region_attributes"]["Object Type"]]
                            
                            x_min = i["shape_attributes"]["x"]
                            y_min = i["shape_attributes"]["y"]
                            x_max = x_min + i["shape_attributes"]["width"]
                            y_max = y_min + i["shape_attributes"]["height"]
                            seg_map[y_min:y_max, x_min:x_max] = mask_value # coordinate system from Annotation Tool and Numpy is different

                            region_names.append(i["region_attributes"]["Object Type"])
                            region_labels.append(mask_value)                           
                        except:
                            pass # do not add these annotations
                     
                    if self.vis:
                       
                        label_names = ["Background", "Button", "Text", "Input Field", "Image"]
                       
                        labelviz_withname1 = imgviz.label2rgb(seg_map, label_names=label_names, font_size=25, image=website_img)
                        seg_map_vis = np.hstack((website_img, labelviz_withname1))
                        full_file_name = os.path.join(self.vis_dir, "vis_gt_{}".format(annotation["filename"]))
                        cv2.imwrite(full_file_name, seg_map_vis)
                    

                self.dataset.append( (annotation["filename"], website_img, seg_map, saliency_map, website_type)  )
        

    def process_eccv_data(self):

        category_count = {}
        for i in self.eccv_categories:
            category_count[str(i)] = 0

        for webpage in self.eccv_data:

            
            # Save imgs for annotations
            web_jpg = webpage[0]
            web_eye_gaze = webpage[1].squeeze(0)[0]
            category = webpage[3]
            
            H, W, _ = web_jpg.shape
            
            web_eye_gaze = web_eye_gaze -  web_eye_gaze.min()
            web_eye_gaze = web_eye_gaze / web_eye_gaze.max()
            web_eye_gaze[:,0] *= W
            web_eye_gaze[:,1] *= H

            category_count[str(i)] += 1
            count = str(category_count[str(i)])

            basename = 'eccv_2015_{}_{}.jpg'.format(category[0], count), web_jpg

            # TODO: grab the region annotations + build segm map
    
    def __len__(self):
        """
        Overwrite the len function to get the number of images in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx:int):
        return self.dataset[idx]

 

   
   

       
        

        
        

