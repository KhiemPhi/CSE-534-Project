import cv2
import numpy as np

import cv2
import numpy as np
from tqdm import tqdm

import imgviz

def GaussianMask(sizex,sizey, sigma=33, center=None,fix=1):
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
    
    
def get_sal_map(fix_arr, H, W):
    heatmap = np.zeros((H,W), np.float32)
    for n_subject in (range(fix_arr.shape[0])):
        heatmap += GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
                                fix_arr[n_subject,2])

    # Normalization
    heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")

    return heatmap

def Fixpos2Densemap(fix_arr, W, H, imgfile, alpha=0.9, threshold=255):
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
        heatmap += GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
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


def generate_mask_from_bbox(regions, seg_map, website_element_dict):

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
    
    return seg_map

def visualize_seg_map(seg_map, img):
    label_names = ["Background", "Button", "Text", "Input Field", "Image"]      
    color_map = np.array([[255,255,255], [255,0,0], [0,255,0], [0,0,255], [255,10,122]])

    labelviz_withname = imgviz.label2rgb(seg_map, label_names=label_names, font_size=25, image=img, colormap=color_map)

    return labelviz_withname