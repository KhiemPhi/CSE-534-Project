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

    def GaussianMask(self, sizex,sizey, sigma=33, center=None,fix=1):
        """
        sizex  : mask width
        sizey  : mask height
        sigma  : gaussian Sd
        center : gaussian mean
        fix    : gaussian max
        return gaussian mask
        """
        x = np.arange(0, sizex, 1, float)
        y = np.arange(0, sizey, 1, float)
        x, y = np.meshgrid(x,y)
        
        if center is None:
            x0 = sizex // 2
            y0 = sizey // 2
        else:
            if np.isnan(center[0])==False and np.isnan(center[1])==False:            
                x0 = center[0]
                y0 = center[1]        
            else:
                return np.zeros((sizey,sizex))

        return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

    def Fixpos2Densemap(self, fix_arr, W, H, imgfile, alpha=0.9, threshold=255):
        """
        fix_arr   : fixation array number of subjects x 3(x,y,fixation)
        width     : output image width
        height    : output image height
        imgfile   : image file (optional)
        alpha     : marge rate imgfile and heatmap (optional)
        threshold : heatmap threshold(0~255)
        return heatmap 
        """
        
        heatmap = np.zeros((H,W), np.float32)
        for n_subject in tqdm(range(fix_arr.shape[0])):
            heatmap += self.GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
                                    fix_arr[n_subject,2])

        # Normalization
        heatmap = heatmap/np.amax(heatmap)
        heatmap = heatmap*255
        heatmap = heatmap.astype("uint8")
        
        if imgfile.any():
        
            # Resize heatmap to imgfile shape 
            h, w, _ = imgfile.shape
            heatmap = cv2.resize(heatmap, (w, h))
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Create mask
            mask = np.where(heatmap<=threshold, 1, 0)
            mask = np.reshape(mask, (h, w, 1))
            mask = np.repeat(mask, 3, axis=2)

            # Marge images
            marge = imgfile*mask + heatmap_color*(1-mask)
            marge = marge.astype("uint8")
            marge = cv2.addWeighted(imgfile, 1-alpha, marge,alpha,0)
            return marge

        else:
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            return heatmap