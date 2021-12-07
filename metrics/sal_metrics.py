
import numpy as np 
import torch

def NSS(saliency_map, fixation_map):
    """"
    normalized scanpath saliency between two different
    saliency maps as the mean value of the normalized saliency map at
    fixation locations.
        Computer NSS score.
        :param saliency_map : predicted saliency map
        :param fixation_map : ground truth saliency map.
        :return score: float : score
    """
    MAP = (saliency_map - fixation_map.mean()) / (fixation_map.std())
    mask = saliency_map.type(torch.bool)
    score =  MAP[mask].mean()

    return score

def CC(saliency_map, saliency_map_gt):
    """
    This finds the linear correlation coefficient between two different
    saliency maps (also called Pearson's linear coefficient).
    score=1 or -1 means the maps are correlated
    score=0 means the maps are completely uncorrelated
    saliencyMap1 and saliencyMap2 are 2 real-valued matrices
        Computer CC score .
        :param saliency_map : first saliency map
        :param saliency_map_gt : second  saliency map.
        :return score: float : score
    """
    if not isinstance(saliency_map, np.ndarray):
        saliency_map = np.array(saliency_map, dtype=np.float32)
    elif saliency_map.dtype != np.float32:
        saliency_map = saliency_map.astype(np.float32)

    if not isinstance(saliency_map_gt, np.ndarray):
        saliency_map_gt = np.array(saliency_map_gt, dtype=np.float32)
    elif saliency_map.dtype != np.float32:
        saliency_map_gt = saliency_map_gt.astype(np.float32)

   
    saliency_map = (saliency_map - saliency_map.mean()) / (saliency_map.std())
    saliency_map_gt = (saliency_map_gt - saliency_map_gt.mean()) / (saliency_map_gt.std())

    score = np.corrcoef(saliency_map.flatten(),saliency_map_gt.flatten())[0][1]

    return score

def entropy(p, dim = -1, keepdim = None):
       return -torch.where(p > 0, p * p.log(), p.new([0.0]))

def KLdiv(saliency_map, saliency_map_gt):
    """
    This finds the KL-divergence between two different saliency maps when
    viewed as distributions: it is a non-symmetric measure of the information
    lost when saliencyMap is used to estimate fixationMap.
        Computer KL-divergence.
        :param saliency_map : predicted saliency map
        :param fixation_map : ground truth saliency map.
        :return score: float : score
    """   

    EPS = np.finfo(np.float32).eps
    # the function will normalize maps before computing Kld   
    score = entropy(saliency_map.flatten() + EPS, saliency_map_gt.flatten() + EPS)
    return score.mean()

def AUC(saliency_map, fixation_map):
    """Computes AUC for given saliency map 'saliency_map' and given
    fixation map 'fixation_map'
    """
    def area_under_curve(predicted, actual, labelset):
        def roc_curve(predicted, actual, cls):
            si = np.argsort(-predicted)
            tp = np.cumsum(np.single(actual[si]==cls))
            fp = np.cumsum(np.single(actual[si]!=cls))
            tp = tp/np.sum(actual==cls)
            fp = fp/np.sum(actual!=cls)
            tp = np.hstack((0.0, tp, 1.0))
            fp = np.hstack((0.0, fp, 1.0))
            return tp, fp
        def auc_from_roc(tp, fp):
            h = np.diff(fp)
            auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
            return auc

        tp, fp = roc_curve(predicted, actual, np.max(labelset))
        auc = auc_from_roc(tp, fp)
        return auc

    fixation_map = (fixation_map>0.7).astype(int)
    salShape = saliency_map.shape
    fixShape = fixation_map.shape

    predicted = saliency_map.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
    actual = fixation_map.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
    labelset = np.arange(2)

    return area_under_curve(predicted, actual, labelset)

def SAUC(saliency_map, fixation_map, shuf_map=np.zeros((480,640)), step_size=.01):
    """
        please cite:  https://github.com/NUS-VIP/salicon-evaluation
        calculates shuffled-AUC score.
        :param salinecy_map : predicted saliency map
        :param fixation_map : ground truth saliency map.
        :return score: int : score
    """

    saliency_map -= np.min(saliency_map)
    fixation_map = np.vstack(np.where(fixation_map!=0)).T

    if np.max(saliency_map) > 0:
        saliency_map = saliency_map / np.max(saliency_map)
   
    Sth = np.asarray([ saliency_map[y-1][x-1] for y,x,gaze in fixation_map ])
    Nfixations = len(fixation_map)

    others = np.copy(shuf_map)
    for y,x in fixation_map:
        others[y-1][x-1] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = shuf_map[ind]
    randfix = saliency_map[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    score = np.trapz(tp,fp)
    return score