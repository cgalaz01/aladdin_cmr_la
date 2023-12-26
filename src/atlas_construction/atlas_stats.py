import os
from typing import List

import numpy as np

import pyvista as pv


def filter_cases(patient_list, patients: bool = False) -> List[str]:
    """
    Filter the patient list based on the patients flag.

    Parameters
    ----------
    patient_list : List[str]
        The list of patient names.
    patients : bool, optional
        Flag indicating whether to include only patients or exclude patients.
        Default is False.

    Returns
    -------
    new_patient_list : List[str]
        The filtered patient list.
    
    """
    new_patient_list = []
    for patient in patient_list:
        if not patients:
            if not patient.lower().startswith('pat'):
                new_patient_list.append(patient)
        else:
            if patient.lower().startswith('pat'):
                new_patient_list.append(patient)

    return new_patient_list  


def load_data(base_path: str, patients: List[str], time_idx: int) -> List[pv.PolyData]:
    """
    Load the cardiac phase mesh for each patient.

    Parameters
    ----------
    base_path : str
        The base path where the patient data is stored.
    patients : List[str]
        The list of patient names.
    time_idx : int
        The index of the desired cardiac phase.

    Returns
    -------
    patient_meshes : List[pv.PolyData]
        The list of loaded patient meshes.
    
    """
    # Read specific cardiac phase mesh across patients
    patient_meshes = []
    for patient in patients:
        patient_path = os.path.join(base_path, patient, 'final_meshes')
        mesh_file = sorted(os.listdir(patient_path))[time_idx]
        file_path = os.path.join(patient_path, mesh_file)
        mesh = pv.read(file_path)
        patient_meshes.append(mesh)    
        
    return patient_meshes


def calculate_metrics(meshes: List[pv.PolyData], atlas_mesh: pv.PolyData) -> pv.PolyData:
    """
    Calculate metrics for each point in the meshes and update the atlas mesh.
    The metrics estimated across the patients are the mean, standard deviation,
    coefficient of variation and covariance.

    Parameters
    ----------
    meshes : List[pv.PolyData]
        The list of patient meshes.
    atlas_mesh : pv.PolyData
        The atlas mesh.

    Returns
    -------
    atlas_mesh : pv.PolyData
        The updated atlas mesh with calculated metrics.
    
    """
    keys = meshes[0].point_data.keys()
    for key in keys:
        values = []
        for mesh_idx in range(len(meshes)):
            values.append(meshes[mesh_idx].point_data[key])
        values = np.asarray(values)
        
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        
        cov = []
        for point_idx in range(values.shape[1]):
            cov.append(np.cov(values[:, point_idx], rowvar=False))    
        cov = np.asarray(cov)
        
        mean_sign = np.less(mean, np.zeros_like(mean)) * -2 + 1
        coef = std / (np.maximum(np.abs(mean), 1e-9) * mean_sign)
        
        new_key = 'mean_' + key
        atlas_mesh.point_data[new_key] = mean
        new_key = 'std_' + key
        atlas_mesh.point_data[new_key] = std
        new_key = 'cov_' + key
        atlas_mesh.point_data[new_key] = cov
        new_key = 'coef_' + key
        atlas_mesh.point_data[new_key] = coef
    
    return atlas_mesh
    

if __name__ == '__main__':
    mesh_folder = '_registration_output'
    use_patients = False
    
    if use_patients:
        atlas_output_folder = '_atlas_stats_patients_output'
    else:
        atlas_output_folder = '_atlas_stats_output'
    
    atlas_path = os.path.join('_atlas_output', 'atlas_mesh.vtk')
    
    # Load patient data
    patients = filter_cases(sorted(os.listdir(mesh_folder)), patients=use_patients)
    mesh_files = sorted(os.listdir(os.path.join(mesh_folder, patients[0], 'final_meshes')))
    
    for k in range(len(mesh_files)):
        mesh_file = mesh_files[k]
        # Reload atlas mesh for each time point
        atlas_mesh = pv.read(atlas_path)
        # Read values across patients
        patient_meshes = load_data(mesh_folder, patients, time_idx=k)
        
        atlas_mesh = calculate_metrics(patient_meshes, atlas_mesh)
        
        # Save mesh to file
        os.makedirs(atlas_output_folder, exist_ok=True)
        atlas_output_file = os.path.join(atlas_output_folder, f'atlas_stats_{k:02d}.vtk')
        atlas_mesh.save(atlas_output_file)
        