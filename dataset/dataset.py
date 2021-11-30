from torch.utils.data import Dataset
import scipy.io
import numpy as np

class WhataboutismDatasetSiamese(Dataset):

    def __init__(self, matlab_path, json_path, imgs_dir, saliency_dir):
        super().__init__()

        #1. Process the Matlab files to get the images + saliency regions
        self.eccv_data = scipy.io.loadmat(matlab_path)["webpages"]

        self.eccv_data =  [ x[0][0][0] for x in self.eccv_data]

        self.eccv_categories = list(np.unique([ x[3] for x in self.eccv_data]))

        self.process_eccv_data()

        #2. Parse the JSON file to get the correct annotations, we build a list of tuple that goes:
        # file_name, img, seg map, saliency map 
        






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

 

   
   

       
        

        
        

