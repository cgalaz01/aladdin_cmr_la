import os
from typing import Any, List, Tuple, Union
import numpy as np

import SimpleITK as sitk
      
from tf.utils import load
from utils.misc import get_file_extension, save_numpy_to_sitk

    
    
def load_files(img_folder_path: str, seg_folder_path: str, folder_seg_path:str,
               file_list: List[str], fixed_idx: int, dg: Any) -> Tuple[Union[sitk.Image, np.ndarray]]:
    """
    Loads the required files for the given fixed image index.

    Parameters
    ----------
    img_folder_path : str
        Path to the patient image folder.
    seg_folder_path : str
        Path to the patient segmentation folder.
    folder_seg_path : str
        Path to the patient non-dilated segmentation folder.
    file_list : List[str]
        The list of all possible files of a patient across the cardiac cycle.
    fixed_idx : int
        The index of the fixed image.
    dg : Any
        The appropriate data loader for the model.

    Returns
    -------
    image_moving_sitk : sitk.Image
        The patient moving image (index=0).
    segmentation_moving_sitk : sitk.Image
        The patient moving segmentation (indewx=0).
    image_fixed_sitk : sitk.Image
        The patient fixed image (index=fixed_idx).
    segmentation_fixed_sitk : sitk.Image
        The patient fixed segmentation (index=fixed_idx).
    contour_fixed : sitk.Image
        The patient fixed non-dilated segmentation (index=fixed_idx).
    contour_moving : sitk.Image
        The patient moving non-dilated segmentation (index=0).
    mask_fixed : np.ndarray
        The left atrium fixed segmentation region (index=fixed_idx).

    """
    moving_idx = 0
    # Get expected file paths
    fixed_file_path = os.path.join(img_folder_path, file_list[fixed_idx])
    moving_file_path = os.path.join(img_folder_path, file_list[moving_idx])
    fixed_seg_file_path = os.path.join(seg_folder_path, file_list[fixed_idx])
    moving_seg_file_path = os.path.join(seg_folder_path, file_list[moving_idx])
    fixed_cont_seg_file_path = os.path.join(folder_seg_path, file_list[moving_idx])
    moving_cont_seg_file_path = os.path.join(folder_seg_path, file_list[fixed_idx])
    
    # Load images
    mask_fixed = dg.to_numpy(sitk.ReadImage(fixed_seg_file_path), int)
    
    image_moving_sitk = sitk.ReadImage(moving_file_path)
    segmentation_moving_sitk = sitk.ReadImage(moving_seg_file_path)
    segmentation_moving_sitk = segmentation_moving_sitk == 1
    
    image_fixed_sitk = sitk.ReadImage(fixed_file_path)
    segmentation_fixed_sitk = sitk.ReadImage(fixed_seg_file_path)
    segmentation_fixed_sitk = segmentation_fixed_sitk == 1
    
    contour_fixed = dg.to_numpy(sitk.ReadImage(fixed_cont_seg_file_path), int)
    contour_moving = dg.to_numpy(sitk.ReadImage(moving_cont_seg_file_path), int)
    
    return (image_moving_sitk, segmentation_moving_sitk, image_fixed_sitk,
            segmentation_fixed_sitk, contour_fixed, contour_moving, mask_fixed)
    

def get_output_paths(patient: str, output_displacement_path: str,
                     output_segmentation_path: str) -> Tuple[str]:
    """
    Returns the output paths where the images will be stored.

    Parameters
    ----------
    patient : str
        The patient name.
    output_displacement_path : str
        The path to the displacement vector fields.
    output_segmentation_path : str
        The path to the segmentations.

    Returns
    -------
    output_full_path : str
        The path to the displacement vector fields.
    output_contour_path : str
        The path to the displacement vector fields only on the contour.
    output_contour_seg_path : str
        The path to the contour transformed segmentation.
    output_mask_seg_path : str
        The path to the masked left atrium segmentation.

    """
    patient_output_path = os.path.join(output_displacement_path, patient)
    output_full_path = os.path.join(patient_output_path, 'full')
    os.makedirs(output_full_path, exist_ok=True)
    output_contour_path = os.path.join(patient_output_path, 'contour')
    os.makedirs(output_contour_path, exist_ok=True)
    
    patient_output_path = os.path.join(output_segmentation_path, patient)
    output_contour_seg_path = os.path.join(patient_output_path, 'contour')
    os.makedirs(output_contour_seg_path, exist_ok=True)
    output_mask_seg_path = os.path.join(patient_output_path, 'mask')
    os.makedirs(output_mask_seg_path, exist_ok=True)
    
    return output_full_path, output_contour_path, output_contour_seg_path, output_mask_seg_path
    
            
def main(model_folder, data_path, model_type, output_type):
    # Base output folders
    output = 'outputs_' + model_folder + '_' + output_type
    output_displacement_path = os.path.join('..', output, 'displacement_field')
    output_segmentation_path = os.path.join('..', output, 'segmentation')
    output_type, file_extension = get_file_extension(output_type)
    
    data_path = os.path.join(data_path, 'train')
    data_seg_path = os.path.join('..', 'data_contour_3d_1_cache', 'train')
    
    patient_list = sorted(os.listdir(os.path.join('..', 'checkpoint', model_folder)))
    if patient_list[0].endswith('.h5'):
        patient_list = sorted(os.listdir(data_path))
    
    dg = load.get_data_loader(model_type)
    
    for patient in patient_list:
        model = load.load_model(model_folder, patient)
        
        img_folder_path = os.path.join(data_path, patient, 'images')
        seg_folder_path = os.path.join(data_path, patient, 'segmentations')
        folder_seg_path = os.path.join(data_seg_path, patient, 'segmentations')
        
        (output_full_path,
         output_contour_path,
         output_contour_seg_path,
         output_mask_seg_path) = get_output_paths(patient, output_displacement_path,
                                                  output_segmentation_path)
        
        file_list = sorted(os.listdir(seg_folder_path))
        for i in range(len(file_list)):
            (image_moving_sitk,
             segmentation_moving_sitk,
             image_fixed_sitk,
             segmentation_fixed_sitk,
             contour_fixed,
             contour_moving,
             mask_fixed) = load_files(img_folder_path, seg_folder_path,
                                      folder_seg_path, file_list, fixed_idx=i,
                                      dg=dg)
                
            data = dg.to_structure(image_moving_sitk,
                                   image_fixed_sitk,
                                   segmentation_moving_sitk,
                                   segmentation_fixed_sitk,
                                   add_batch_axis=True,
                                   align=False)
            
            results = model.predict(data[0])
            if model_type == 'vxm' or model_type == 'vxmoverlay' or model_type == 'vxmseg':
                vector_idx = 1
            else:
                vector_idx = 4
                
            displacement_field = np.squeeze(results[vector_idx])
            # Convert from image to physical space
            spacing = image_moving_sitk.GetSpacing()
            for axis in range(len(spacing)):
                displacement_field[..., axis] *= spacing[axis]
            # Negate displacement field - currently transformation from fix to moving
            # rather than moving to fixed
            displacement_field = -displacement_field
            
            
            # Save actual target segmentations
            file_path = os.path.join(output_contour_seg_path, '{:02d}.{}'.format(i, file_extension))
            if output_type == 0:
                np.save(file_path, contour_fixed)
            elif output_type == 1:
                save_numpy_to_sitk(contour_fixed.astype(int), segmentation_moving_sitk,
                                   file_path, is_displacement_field=False)
                
            
            file_path = os.path.join(output_mask_seg_path, '{:02d}.{}'.format(i, file_extension))
            if output_type == 0:
                np.save(file_path, mask_fixed)
            elif output_type == 1:
                save_numpy_to_sitk(mask_fixed.astype(int), segmentation_moving_sitk,
                                   file_path, is_displacement_field=False)
            
            # Save displacement fields
            file_path = os.path.join(output_full_path, '{:02d}.{}'.format(i, file_extension))
            if output_type == 0:
                np.save(file_path, displacement_field)
            elif output_type == 1:
                save_numpy_to_sitk(displacement_field, segmentation_moving_sitk,
                                   file_path, is_displacement_field=True)
            
            
            displacement_field = contour_moving[..., np.newaxis] * displacement_field
            file_path = os.path.join(output_contour_path, '{:02d}.{}'.format(i, file_extension))
            if output_type == 0:
                np.save(file_path, displacement_field)
            elif output_type == 1:
                save_numpy_to_sitk(displacement_field, segmentation_moving_sitk,
                                   file_path, is_displacement_field=True)        
            
            