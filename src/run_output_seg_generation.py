import os

import numpy as np

import SimpleITK as sitk

from tf.utils import load
from utils.segmentations import select_largest_region, remove_small_noise, median_filter
from utils.misc import get_file_extension, save_numpy_to_sitk

            
            
def main(model_folder, data_path, output_type):
    output = 'outputs_seg_' + model_folder + '_' + output_type
    output_type, file_extension = get_file_extension(output_type)
    
    output_segmentation_path = os.path.join('..', output, 'segmentation')
    
    
    data_path = os.path.join(data_path, 'train')
    
    patient_list = sorted(os.listdir(os.path.join('..', 'checkpoint', model_folder)))
    
    
    dg = load.get_data_loader('aladdin_s')
    
    for patient in patient_list:
        model = load.load_seg_model(model_folder, patient)
        
        img_folder_path = os.path.join(data_path, patient, 'images')
        seg_folder_path = os.path.join(data_path, patient, 'segmentations')
        
        
        patient_output_path = os.path.join(output_segmentation_path, patient)
        patient_output_path = os.path.join(patient_output_path, 'segmentations')
        os.makedirs(patient_output_path, exist_ok=True)
        
        file_list = sorted(os.listdir(seg_folder_path))
        for i in range(len(file_list)):
            
                
            img_file_path = os.path.join(img_folder_path, file_list[i])
            seg_file_path = os.path.join(seg_folder_path, file_list[i])
            
            image_sitk = sitk.ReadImage(img_file_path)
            segmentation_sitk = sitk.ReadImage(seg_file_path)
            segmentation_sitk = segmentation_sitk == 1            
            
            data = dg.to_structure(image_sitk,
                                   segmentation_sitk,
                                   add_batch_axis=True)
            
            results = model.predict(data[0])
            results_seg = np.squeeze(results)
            results_seg = (results_seg >= 0.5).astype(int)
            results_seg = select_largest_region(results_seg)
            results_seg = remove_small_noise(results_seg)
            results_seg = median_filter(results_seg)
            
            # Save predicted segmentations
            file_path = os.path.join(patient_output_path, '{:02d}.{}'.format(i, file_extension))
            if output_type == 0:
                np.save(file_path, results_seg)
            elif output_type == 1:
                save_numpy_to_sitk(results_seg.astype(int), segmentation_sitk,
                                   file_path, is_displacement_field=False)
        
            
            