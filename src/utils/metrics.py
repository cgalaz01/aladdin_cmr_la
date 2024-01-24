from typing import Dict, Optional, Union, Tuple

import numpy as np
from scipy import ndimage

import voxelmorph as vxm


def convolve_var(array: np.ndarray, kernel_size: Union[int, Tuple[int]]) -> np.ndarray:
    if isinstance(kernel_size, int):
        kernel_size = np.asarray([kernel_size] * array.ndim)
        
    def convolve_function(window: np.ndarray) -> float:
        # Check if center element of window is NaN
        window = window.reshape(kernel_size)
        if np.take(window, np.ravel_multi_index((kernel_size) // 2, kernel_size)) == np.nan:
            return np.nan
        
        var = np.nanvar(window)
        return var
        
    f = convolve_function
    return ndimage.generic_filter(array, f, size=kernel_size, mode='constant',
                                  cval=np.nan, output=float, origin=0)


def displacement_field_magnitude(displacement_field: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(displacement_field[..., 0]**2 +
                        displacement_field[..., 1]**2 +
                        displacement_field[..., 2]**2)
    
    return magnitude


def jacobian_determinant(displacement_field: np.ndarray) -> np.ndarray:
    jd = vxm.py.utils.jacobian_determinant(displacement_field)
    return jd


def jacobian_determinant_stats(displacement_field: np.ndarray,
                               segmentation: Optional[np.ndarray] = None,
                               spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
        
    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
        
    jd = jacobian_determinant(displacement_field)
        
    if not segmentation is None:
        jd[segmentation < 0.5] = np.nan
    
    jd_mean = np.nanmean(jd)
    jd_median = np.nanmedian(jd)
    jd_std = np.nanstd(jd)
    jd_min = np.nanmin(jd)
    jd_max = np.nanmax(jd)
    
    return {'mean': jd_mean,
            'std': jd_std,
            'median': jd_median,
            'min': jd_min,
            'max': jd_max,
            'raw': jd}


def jacobian_spatial_stats(displacement_field: np.ndarray,
                           segmentation: Optional[np.ndarray] = None,
                           spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
    
    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
    
    jd = jacobian_determinant(displacement_field)
    
    if not segmentation is None:
        jd[segmentation < 0.5] = np.nan
            
    jd_var = convolve_var(jd, kernel_size=3)
    
    jd_var_mean = np.nanmean(jd_var)
    jd_var_median = np.nanmedian(jd_var)
    jd_var_std = np.nanstd(jd_var)
    jd_var_min = np.nanmin(jd_var)
    jd_var_max = np.nanmax(jd_var)
    
    return {'mean': jd_var_mean,
            'std': jd_var_std,
            'median': jd_var_median,
            'min': jd_var_min,
            'max': jd_var_max,
            'raw': jd_var}
    

def jacobian_temporal_stats(displacement_field: np.ndarray,
                            prev_displacement_field: np.ndarray,
                            segmentation: Optional[np.ndarray] = None,
                            spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim

    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        prev_displacement_field = prev_displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
            prev_displacement_field[..., i] *= spacing[i]

    jd = jacobian_determinant(displacement_field)
    prev_jd = jacobian_determinant(prev_displacement_field)
    
    if not segmentation is None:
        jd[segmentation < 0.5] = np.nan
        prev_jd[segmentation < 0.5] = np.nan
        
    jd_t = np.concatenate([np.expand_dims(prev_jd, axis=-1),
                            np.expand_dims(jd, axis=-1)], axis=-1)
    _, _, _, jd_dt = np.gradient(jd_t)
    jd_t = jd_t[..., 1]
    
    jd_dt_mean = np.nanmean(jd_dt)
    jd_dt_median = np.nanmedian(jd_dt)
    jd_dt_std = np.nanstd(jd_dt)
    jd_dt_min = np.nanmin(jd_dt)
    jd_dt_max = np.nanmax(jd_dt)
    
    return {'mean': jd_dt_mean,
            'std': jd_dt_std,
            'median': jd_dt_median,
            'min': jd_dt_min,
            'max': jd_dt_max,
            'raw': jd_dt}

    
def magnitude_stats(displacement_field: np.ndarray,
                    segmentation: Optional[np.ndarray] = None,
                    spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
        
    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
        
    magnitude = displacement_field_magnitude(displacement_field)
    
    if not segmentation is None:
        magnitude[segmentation < 0.5] = np.nan
    
    mag_mean = np.nanmean(magnitude)
    mag_median = np.nanmedian(magnitude)
    mag_std = np.nanstd(magnitude)
    mag_min = np.nanmin(magnitude)
    mag_max = np.nanmax(magnitude)
    
    return {'mean': mag_mean,
            'std': mag_std,
            'median': mag_median,
            'min': mag_min,
            'max': mag_max,
            'raw': magnitude}


def magnitude_spatial_stats(displacement_field: np.ndarray,
                            segmentation: Optional[np.ndarray] = None,
                            spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
        
    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
        
    magnitude = displacement_field_magnitude(displacement_field)
    
    if not segmentation is None:
        magnitude[segmentation < 0.5] = np.nan
        
    mag_var = convolve_var(magnitude, kernel_size=3)
    
    mag_var_mean = np.nanmean(mag_var)
    mag_var_median = np.nanmedian(mag_var)
    mag_var_std = np.nanstd(mag_var)
    mag_var_min = np.nanmin(mag_var)
    mag_var_max = np.nanmax(mag_var)
    
    return {'mean': mag_var_mean,
            'std': mag_var_std,
            'median': mag_var_median,
            'min': mag_var_min,
            'max': mag_var_max,
            'raw': mag_var}
    

def magnitude_temporal_stats(displacement_field: np.ndarray,
                             prev_displacement_field: np.ndarray,
                             segmentation: Optional[np.ndarray] = None,
                             spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
    
    if np.any(spacing != 1):
        displacement_field = displacement_field.copy()
        prev_displacement_field = prev_displacement_field.copy()
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
            prev_displacement_field[..., i] *= spacing[i]
        
        
    magnitude = displacement_field_magnitude(displacement_field)
    prev_mag = displacement_field_magnitude(prev_displacement_field)
    
    if not segmentation is None:
        magnitude[segmentation < 0.5] = np.nan
        prev_mag[segmentation < 0.5] = np.nan
    
    mag_t = np.concatenate([np.expand_dims(prev_mag, axis=-1),
                            np.expand_dims(magnitude, axis=-1)], axis=-1)
    _, _, _, mag_dt = np.gradient(mag_t)
    mag_dt = mag_dt[..., 1]
    
    mag_dt_mean = np.nanmean(mag_dt)
    mag_dt_median = np.nanmedian(mag_dt)
    mag_dt_std = np.nanstd(mag_dt)
    mag_dt_min = np.nanmin(mag_dt)
    mag_dt_max = np.nanmax(mag_dt)
    
    return {'mean': mag_dt_mean,
            'std': mag_dt_std,
            'median': mag_dt_median,
            'min': mag_dt_min,
            'max': mag_dt_max,
            'raw': mag_dt}


def displacement_field_spatial_stats(displacement_field: np.ndarray,
                                     segmentation: Optional[np.ndarray] = None,
                                     spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
        
    if not segmentation is None:
        #segmentation = np.expand_dims(segmentation, axis=-1)
        displacement_field = displacement_field.copy()
        for i in range(displacement_field.shape[-1]):
            displacement_field[..., i][segmentation < 0.5] = np.nan
         
    if np.any(spacing != 1):
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
        
    df_var = np.zeros_like(displacement_field[..., 0])
    for i in range(displacement_field.shape[-1]):
        df_var += convolve_var(displacement_field[..., i], kernel_size=3)
    df_var /= displacement_field.shape[-1]
    
    df_var_mean = np.nanmean(df_var)
    df_var_median = np.nanmedian(df_var)
    df_var_std = np.nanstd(df_var)
    df_var_min = np.nanmin(df_var)
    df_var_max = np.nanmax(df_var)
    
    return {'mean': df_var_mean,
            'std': df_var_std,
            'median': df_var_median,
            'min': df_var_min,
            'max': df_var_max,
            'raw': df_var}
    

def displacement_field_temporal_stats(displacement_field: np.ndarray,
                                      prev_displacement_field: np.ndarray,
                                      segmentation: Optional[np.ndarray] = None,
                                      spacing: Union[float, Tuple[float]] = 1) -> Dict[str, float]:
    if np.isscalar(spacing):
        spacing = [spacing] * displacement_field[..., 0].ndim
    
    if not segmentation is None:
        #segmentation = np.expand_dims(segmentation, axis=-1)
        displacement_field = displacement_field.copy()
        prev_displacement_field = prev_displacement_field.copy()
        for i in range(displacement_field.shape[-1]):
            displacement_field[segmentation < 0.5] = np.nan
            prev_displacement_field[segmentation < 0.5] = np.nan
        
    if np.any(spacing != 1):
        for i in range(len(spacing)):
            displacement_field[..., i] *= spacing[i]
            prev_displacement_field[..., i] *= spacing[i]
        
    df_dt = np.zeros_like(displacement_field[..., 0])
    for i in range(displacement_field.shape[-1]):
        df_t = np.concatenate([np.expand_dims(prev_displacement_field[..., i], axis=-1),
                                np.expand_dims(displacement_field[..., i], axis=-1)], axis=-1)
    
        _, _, _, df_dt_axis = np.gradient(df_t)
        df_dt += df_dt_axis[..., 1]
    df_dt /= displacement_field.shape[-1]
    
    df_dt_mean = np.nanmean(df_dt)
    df_dt_median = np.nanmedian(df_dt)
    df_dt_std = np.nanstd(df_dt)
    df_dt_min = np.nanmin(df_dt)
    df_dt_max = np.nanmax(df_dt)
    
    return {'mean': df_dt_mean,
            'std': df_dt_std,
            'median': df_dt_median,
            'min': df_dt_min,
            'max': df_dt_max,
            'raw': df_dt}

