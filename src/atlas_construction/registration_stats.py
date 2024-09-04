import os

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis

import pyvista as pv


def probability_in_distribution(values: np.ndarray, mean: float,
                                std: float) -> np.ndarray:
    """
    Calculate the probability of values being in the distributionusing the cumulative distribution function (CDF).

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    mean : float
        Mean value.
    std : float
        Standard deviation.

    Returns
    -------
    probability : np.ndarray
        Array of probabilities.

    """
    z_score = (values - mean) / np.maximum(std, 1e-9)
    probability = stats.norm.cdf(z_score)
    return probability


def get_1d_mahalanobis_distance(values: np.ndarray, mean: float,
                                std: float) -> np.ndarray:
    """
    Calculate the Mahalanobis distance for 1-dimensional data. This is equivalent
    to the z-score.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    mean : float
        Mean value.
    std : float
        Standard deviation.

    Returns
    -------
    distance : np.ndarray
        Array of Mahalanobis distances.

    """
    distance = np.abs(values - mean) / np.maximum(std, 1e-9)
    return distance


def get_nd_mahalanobis_distance(values: np.ndarray, mean: float,
                                cov: np.ndarray) -> np.ndarray:
    """
    Calculate the Mahalanobis distance for n-dimensional data.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    mean : float
        Mean value.
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    distance : np.ndarray
        Array of Mahalanobis distances.

    """
    # Reshape covariance matrix back to 2D
    cov_size = int(np.sqrt(cov.shape[0]))
    cov = cov.reshape((cov_size, cov_size))
    inv_cov = np.linalg.pinv(cov)
    distance = mahalanobis(values, mean, inv_cov)
    return distance


def calculate_mahalanobis_distance(values: np.ndarray, mean: np.ndarray,
                                   cov: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Calculate the Mahalanobis distance for both 1-dimensional and n-dimensional data.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    mean : np.ndarray
        Array of mean values.
    cov : np.ndarray
        Array of covariance matrices.
    std : np.ndarray
        Array of standard deviations.

    Returns
    -------
    distances : np.ndarray
        Array of Mahalanobis distances.

    """
    distances = np.zeros((values.shape[0]), dtype=float)
    for point_idx in range(values.shape[0]):
        if len(values.shape) == 1 or values.shape[-1] == 1:
            distances[point_idx] = get_1d_mahalanobis_distance(values[point_idx], 
                                                               mean[point_idx],
                                                               std[point_idx])
        else:
            distances[point_idx] = get_nd_mahalanobis_distance(values[point_idx],
                                                               mean[point_idx],
                                                               cov[point_idx])

    
    return distances


def calculate_metrics(patient_mesh: pv.PolyData, atlas_mesh: pv.PolyData) -> pv.PolyData:
    """
    Calculate metrics for the patient mesh based on the atlas mesh.

    Parameters
    ----------
    patient_mesh : pv.PolyData
        The patient mesh.
    atlas_mesh : pv.PolyData
        The atlas mesh.

    Returns
    -------
    patient_mesh : pv.PolyData
        The patient mesh with calculated metrics.

    """
    keys = patient_mesh.point_data.keys()
    for key in keys:
        atlas_mean_key = 'mean_' + key
        atlas_std_key = 'std_' + key
        atlas_cov_key = 'cov_' + key
        difference = patient_mesh.point_data[key] - atlas_mesh.point_data[atlas_mean_key]
        patient_mesh.point_data['diff_' + key] = difference

        mahalanobis_distance = calculate_mahalanobis_distance(patient_mesh.point_data[key],
                                                              atlas_mesh.point_data[atlas_mean_key],
                                                              atlas_mesh.point_data[atlas_cov_key],
                                                              atlas_mesh.point_data[atlas_std_key])
        
        patient_mesh.point_data['mah_' + key] = mahalanobis_distance
        
        probability = probability_in_distribution(patient_mesh.point_data[key],
                                                  atlas_mesh.point_data[atlas_mean_key],
                                                  atlas_mesh.point_data[atlas_std_key])
        patient_mesh.point_data['prob_' + key] = probability
        
    return patient_mesh


if __name__ == '__main__':
    mesh_folder = '_registration_output'
    atlas_stats_folder = '_atlas_stats_output'    
    output_folder = '_registration_stats_output'
    
    patients = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.vtk')])
    atlas_files = sorted([f for f in os.listdir(atlas_stats_folder) if f.endswith('.vtk')])
    
    for patient in patients:
        print('Processing: ' + patient)
        mesh_files = sorted(os.listdir(os.path.join(mesh_folder, patient)))
        for i in range(len(mesh_files)):    
            patient_mesh = pv.read(os.path.join(mesh_folder, patient, mesh_files[i]))
            atlas_mesh = pv.read(os.path.join(atlas_stats_folder, atlas_files[i]))
            
            patient_mesh = calculate_metrics(patient_mesh, atlas_mesh)
            
            patient_output_folder = os.path.join(output_folder, patient)
            os.makedirs(patient_output_folder, exist_ok=True)
            patient_mesh.save(os.path.join(patient_output_folder, f'mesh_stats_{i:02d}.vtk'))