import numpy as np
from PIL import Image
import scipy.io
import scipy.ndimage


def auc_calc_score(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    """
    Computer AUC score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    # Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])
    y,x = np.where(gtsAnn==1)
    tmp = []
    for i in range(x.shape[0]):
        tmp.append(salMap[y[i],x[i]])
    Sth = np.array(tmp)

    Nfixations = len(x)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

def auc_compute_score(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts'):
    """
    Computes AUC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param res : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert(len(gts) == len(sals))
    score = []
    for i in range(len(sals)):

        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)
        
        mat = scipy.io.loadmat(gts[i])
        fixations = mat[fxt_field_in_mat]
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1],image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0],salmap.shape[1])
        height_fx,width_fx = (fixations.shape[0],fixations.shape[1])
        salmap = scipy.ndimage.zoom(salmap, 
            (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap / np.max(salmap)

        score.append(auc_calc_score(fixations,salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)


def cc_calc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def cc_compute_score(sals, gts, image_size=(480, 640), sigma=-1.0):
    """
    Computes CC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert(len(gts) == len(sals))
    score = []
    for i in range(len(sals)):
        
        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)
            
        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)

        height_sal, width_sal = (salmap.shape[0],salmap.shape[1])
        height_fx,width_fx = (gtmap.shape[0],gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap, 
                (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, 
                (float(image_size[0])/height_sal, float(image_size[1])/width_sal), order=3)
            gtmap = scipy.ndimage.zoom(gtmap, 
                (float(image_size[0])/height_fx, float(image_size[1])/width_fx), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap /np.max(salmap)

        score.append(cc_calc_score(gtmap,salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)



def nss_calc_score(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    # return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

    # y,x = np.where(gtsAnn==1)
    # scores = []
    # for i in range(x.shape[0]):
    #     scores.append(salMap[y[i],x[i]])
    # return np.nanmean(np.array(scores))
    return np.sum(salMap*gtsAnn)/np.sum(gtsAnn)

def nss_compute_score(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts'):
    """
    Computes NSS score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : saliency map predictions with "image name" key and ndarray as values
    :param image_size: [height, width]
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert(len(gts) == len(sals))

    score = []
    for i in range(len(sals)):
        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])
        fixations = mat[fxt_field_in_mat]
        fixations = fixations.astype(np.bool)
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1],image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0],salmap.shape[1])
        if image_size is None:
            height_fx,width_fx = (fixations.shape[0],fixations.shape[1])
            salmap = scipy.ndimage.zoom(salmap, 
                (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        else:
            height_fx,width_fx = (image_size[0],image_size[1])
            salmap = scipy.ndimage.zoom(salmap, 
                (float(image_size[0])/height_sal, float(image_size[1])/width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap /np.max(salmap)

        score.append(nss_calc_score(fixations,salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)