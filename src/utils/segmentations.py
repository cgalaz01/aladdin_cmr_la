import numpy as np

from skimage import measure, morphology
from scipy import ndimage


def select_largest_region(segmentation: np.ndarray) -> np.ndarray:
    largest_seg = np.zeros_like(segmentation)
    for i in range(segmentation.shape[-1]):
        labels, num = measure.label(segmentation[..., i].astype(bool), background=0, connectivity=2,
                                    return_num=True)
        if num == 0:
            continue
        elif num == 1:
            largest_seg[..., i] = segmentation[..., i]
        
        unique, counts = np.unique(labels, return_counts=True)
        unique = unique[1:]
        counts = counts[1:]
        max_label = unique[counts == counts.max()][0]
        largest_region = labels == max_label
        largest_seg[..., i] = largest_region.astype(segmentation.dtype)
    
    return largest_seg


def median_filter(segmentation: np.ndarray) -> np.ndarray:
    smoothed_segmentation = ndimage.median_filter(segmentation, size=(5, 5, 3))
    smoothed_segmentation = (smoothed_segmentation >= 0.5).astype(smoothed_segmentation.dtype)
    return smoothed_segmentation


def remove_small_noise(segmentation: np.ndarray) -> np.ndarray:
    
    for i in range(segmentation.shape[-1]):
        segmentation[..., i] = morphology.area_closing(segmentation[..., i],
                                                       area_threshold=64,
                                                       connectivity=1)
        
        segmentation[..., i] = morphology.remove_small_objects(segmentation[..., i],
                                                               min_size=32,
                                                               connectivity=1)
        
    return segmentation
