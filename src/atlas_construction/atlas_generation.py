import os
import math

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure


import pyvista as pv
import SimpleITK as sitk


__LA_LABEL = 1
__LV_LABEL = 2
__RV_LABEL = 3


def filter_cases(patient_list: List[str]) -> List[str]:
    """
    Filter the patient list to exclude cases that start with 'pat'
    resulting to only healthy cases.

    Parameters
    ----------
    patient_list : List[str]
        The list of patient names.

    Returns
    -------
    healthy_cases : List[str]
        The filtered list of patient names.

    """
    new_patient_list = []
    
    for patient in patient_list:
        if not patient.lower().startswith('pat'):
            new_patient_list.append(patient)

    return new_patient_list    

    

def load_reference_data(base_path: str, patients: List[str]) -> Tuple[sitk.Image, sitk.Image]:
    """
    Load one of the patient cases (first case) as the reference case for the
    atlas generation.

    Parameters
    ----------
    base_path : str
        The base path where the patient data is stored.
    patients : List[str]
        The list of patient names.

    Returns
    -------
    reference_image : sitk.Image
        The reference image used in the atlas generation.
    reference_segmentation : sitk.Image
        The reference segmentation used in the atlas generation.

    """
    reference_case = patients[0]
    image_path = os.path.join(base_path, reference_case, 'images')
    segmentation_path = os.path.join(base_path, reference_case, 'segmentations')
    
    file_path = os.path.join(image_path, '00.nii.gz')
    reference_image = sitk.ReadImage(file_path)
    file_path = os.path.join(segmentation_path, '00.nii.gz')
    reference_segmentation = sitk.ReadImage(file_path)
    
    return reference_image, reference_segmentation


def load_patient_data(base_path: str, patients: List[str], target_image: sitk.Image,
                      target_segmentation: sitk.Image) -> Tuple[List[sitk.Image], List[sitk.Image]]:
    """
    Load the patient data for registration.

    Parameters
    ----------
    base_path : str
        The base path where the patient data is stored.
    patients : List[str]
        The list of patient names.
    target_image : sitk.Image
        The registration target image to be used as a reference.
    target_segmentation : sitk.Image
        The registration target segmentation to be used as a reference.

    Returns
    -------
    moving_images : List[sitk.Image]
        The list of moving images for registration.
    moving_segmentations : List[sitk.Image]
        The list of moving segmentations for registration.

    """
    moving_images = []
    moving_segmentations = []
    
    lazy_physical_alignment = True
    for patient in patients:
        image_path = os.path.join(base_path, patient, 'images')
        segmentation_path = os.path.join(base_path, patient, 'segmentations')

        file_path = os.path.join(image_path, '00.nii')
        moving_image = sitk.ReadImage(file_path)
        file_path = os.path.join(segmentation_path, '00.nii')
        moving_segmentation = sitk.ReadImage(file_path)\
        
        if lazy_physical_alignment:
            moving_image.CopyInformation(target_image)
            moving_segmentation.CopyInformation(target_segmentation)
        else:
            moving_image = sitk.Resample(moving_image, target_image, sitk.Transform(),
                                         sitk.sitkLinear, 0.0, moving_image.GetPixelID())
            moving_segmentation = sitk.Resample(moving_segmentation, target_segmentation,
                                                sitk.Transform(), sitk.sitkNearestNeighbor,
                                                0, moving_segmentation.GetPixelID())
        
        moving_images.append(moving_image)
        moving_segmentations.append(moving_segmentation)
        
    return moving_images, moving_segmentations


def rigid_register(target_image: sitk.Image, target_segmentation: sitk.Image,
                   moving_image: sitk.Image, moving_segmentation: sitk.Image) -> sitk.Transform:
    """
    Perform rigid registration between the moving image and the target image
    using the centroids of the left atrium, right atrium, and left ventricle
    from the target segmentation and moving segmentation.

    Parameters
    ----------
    target_image : sitk.Image
        The target image to register.
    target_segmentation : sitk.Image
        The segmentation of the target image.
    moving_image : sitk.Image
        The moving image to register.
    moving_segmentation : sitk.Image
        The segmentation of the moving image.

    Returns
    -------
    rigid_transformation : sitk.Transform
        The rigid transformation that aligns the moving image with the target
        image.

    """
    # Get the center of mass of each of the labels in the images
    target_seg = sitk.GetArrayFromImage(target_segmentation)
    target_seg = np.swapaxes(target_seg, 0, -1)
    target_seg = target_seg.astype(int)
    moving_seg = sitk.GetArrayFromImage(moving_segmentation)
    moving_seg = np.swapaxes(moving_seg, 0, -1)
    moving_seg = moving_seg.astype(int)
    
    target_labels = np.unique(target_seg)
    moving_labels = np.unique(moving_seg)
    joint_labels = np.union1d(target_labels, moving_labels)
    joint_labels = np.argwhere(joint_labels).flatten()
    
    target_coordinates = []
    moving_coordinates = []
    
    for i in joint_labels:
        sc = ndimage.measurements.center_of_mass(target_seg == i)
        sc = np.asarray(sc).round().astype(int)
        
        mc = ndimage.measurements.center_of_mass(moving_seg == i)
        mc = np.asarray(mc).round().astype(int)
        
        target_coordinates.append(sc)
        moving_coordinates.append(mc)
        
    target_coordinates = np.asarray(target_coordinates)
    moving_coordinates = np.asarray(moving_coordinates)
    
    # Perform translation and rotation around the long-axis of the heart
    # transformation
    
    # Normalize coordinates
    # 1 - Left atrium
    # 2 - Left ventricle
    # 3 - Right ventricle
    # Make Orthogonal - moving_image
    # x,y-axis same for left atrium and left ventricle
    moving_coordinates[0][0] = moving_coordinates[1][0]
    moving_coordinates[0][1] = moving_coordinates[1][1]
    # Same z-axis for left and right ventricle
    moving_coordinates[2][2] = moving_coordinates[1][2]
    
    # Make Orthogonal and roughly align translation - moving_image
    # x,y-axis same for left atrium and left ventricle
    target_coordinates[0][0] = target_coordinates[1][0]
    target_coordinates[0][1] = target_coordinates[1][1]
    # Same z-axis for all points
    target_coordinates[0][2] = moving_coordinates[0][2]
    target_coordinates[1][2] = moving_coordinates[1][2]
    target_coordinates[2][2] = moving_coordinates[2][2]
    
    # Reduce the distance of the target coordinate of the Right Ventricle
    # to match that of the moving
    dx = (moving_coordinates[2][0] - moving_coordinates[1][0])
    dy = (moving_coordinates[2][1] - moving_coordinates[1][1])

    distance_moving = math.sqrt(dx**2 + dy**2)

    dx = (target_coordinates[2][0] - target_coordinates[1][0])
    dy = (target_coordinates[2][1] - target_coordinates[1][1])

    distance_target = math.sqrt(dx**2 + dy**2)

    angle = math.atan2(dy, dx)
    a = math.cos(angle) * (distance_target - distance_moving)
    b = math.sin(angle) * (distance_target - distance_moving)

    x_a = target_coordinates[2][0] - a
    y_b = target_coordinates[2][1] - b
    target_coordinates[2][0] = round(x_a)
    target_coordinates[2][1] = round(y_b)
    
    target_image_points_flat = [c for p in target_coordinates.astype(float) for c in p]    
    moving_image_points_flat = [c for p in moving_coordinates.astype(float) for c in p]
    rigid_transformation = sitk.VersorRigid3DTransform(sitk.LandmarkBasedTransformInitializer(
        sitk.VersorRigid3DTransform(),
        target_image_points_flat,
        moving_image_points_flat,
        referenceImage=target_segmentation,
        numberOfControlPoints=3))
    
    
    target_segmentation_la = (target_seg == 1).astype(int)
    moving_segmentation_la = (moving_seg == 1).astype(int)
    
    target_segmentation_la = np.swapaxes(target_segmentation_la, 0, -1)
    target_segmentation_la = sitk.GetImageFromArray(target_segmentation_la)
    target_segmentation_la.CopyInformation(target_segmentation)
    
    moving_segmentation_la = np.swapaxes(moving_segmentation_la, 0, -1)
    moving_segmentation_la = sitk.GetImageFromArray(moving_segmentation_la)
    moving_segmentation_la.CopyInformation(moving_segmentation)
    
    rigid_transformation = sitk.CenteredTransformInitializer(target_segmentation_la, 
                                                             moving_segmentation_la, 
                                                             rigid_transformation, 
                                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    return rigid_transformation
        

def affine_register(target_segmentation: sitk.Image, moving_segmentation: sitk.Image,
                    initial_transformation: sitk.Transform) -> sitk.Transform:
    """
    Perform affine registration between the moving segmentation and the target
    segmentation.

    Parameters
    ----------
    target_segmentation : sitk.Image
        The target segmentation image.
    moving_segmentation : sitk.Image
        The moving segmentation image.
    initial_transformation : sitk.Transform
        The initial transformation to be used for registration.

    Returns
    -------
    affine_transform : sitk.Transform
        The affine transformation that alines the moving segmentation with the
        target segmentation.

    """
    target_segmentation_la = sitk.GetArrayFromImage(target_segmentation)
    target_segmentation_la = (target_segmentation_la == __LA_LABEL).astype(float)
    target_segmentation_la = sitk.GetImageFromArray(target_segmentation_la)
    target_segmentation_la.CopyInformation(target_segmentation)
    
    moving_segmentation_la = sitk.GetArrayFromImage(moving_segmentation)
    moving_segmentation_la = (moving_segmentation_la == __LA_LABEL).astype(float)
    moving_segmentation_la = sitk.GetImageFromArray(moving_segmentation_la)
    moving_segmentation_la.CopyInformation(moving_segmentation)

    optimized_transform = sitk.AffineTransform(3)
    optimized_transform = sitk.CenteredTransformInitializer(target_segmentation,
                                                            moving_segmentation,
                                                            optimized_transform)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                      numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10,
                                                      estimateLearningRate=registration_method.EachIteration)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetMovingInitialTransform(initial_transformation)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0, 0])

    affine_transform = registration_method.Execute(target_segmentation_la, moving_segmentation_la)
    affine_transform = sitk.CompositeTransform([initial_transformation, affine_transform])
        
    return affine_transform


def affine_register_cases(moving_images: List[sitk.Image], moving_segmentations: List[sitk.Image],
                          target_image: sitk.Image, target_segmentation: sitk.Image) -> List[sitk.Transform]:
    """
    Perform affine registration for a list of moving images and segmentations
    to a target image and segmentation.

    Parameters
    ----------
    moving_images : List[sitk.Image]
        List of moving images to be registered.
    moving_segmentations : List[sitk.Image]
        List of moving segmentations corresponding to the moving images.
    target_image : sitk.Image
        Target image to which the moving images will be registered.
    target_segmentation : sitk.Image
        Target segmentation to which the moving segmentations will be registered.

    Returns
    -------
    affine_transformations : List[sitk.Transform]
        List of affine transformations representing the registration of moving
        images and segmentations to the target image and segmentation.

    """
    affine_transformations = []
    for i in range(len(moving_images)):
        moving_image = moving_images[i]
        moving_segmentation = moving_segmentations[i]
        
        # Landmark rigid registration - make sure orientation is correct
        rigid_transformation = rigid_register(target_image, target_segmentation,
                                              moving_image, moving_segmentation)
        
        affine_transformation = affine_register(target_segmentation,
                                                moving_segmentation,
                                                rigid_transformation)
        
        affine_transformations.append(affine_transformation)
        
    return affine_transformations


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html
def smooth_and_resample(image: sitk.Image, shrink_factors: List[float],
                        smoothing_sigmas: List[float], sampling: int = sitk.sitkLinear) -> sitk.Image:
    """
    Smooths and resamples the input image using Gaussian smoothing and resampling.

    Parameters
    ----------
    image : sitk.Image
        The image to be resampled.
    shrink_factors : List[float]
        The shrink factors for resampling. Each factor should be greater than one.
    smoothing_sigmas : List[float]
        The sigmas for Gaussian smoothing. Each sigma should be in physical
        units, not pixels.
    sampling : int, optional
        The sampling method for resampling. The default is sitk.sitkLinear.

    Returns
    -------
    smoothed_resampled_image : sitk.Image
        The resulting image after smoothing and resampling.

    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    if all(s != 0 for s in smoothing_sigmas):
        smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
    else:
        smoothed_image = image
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sampling, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())


def select_la(image: sitk.Image, segmentation: Optional[sitk.Image] = None) -> sitk.Image:
    """
    Selects the left atrium (LA) region from the input image or segmentation.

    Parameters
    ----------
    image : sitk.Image
        The input image or segmentation.
    segmentation : sitk.Image, optional
        The segmentation mask of the LA. If provided, the function will mask out
        anything outside of the LA based on the segmentation. If not provided,
        the function assumes that the input image is a segmentation and directly
        selects the LA region.

    Returns
    -------
    selected_image : sitk.Image
        The selected image with only the LA region.

    """
    temp = sitk.GetArrayFromImage(image)
    if segmentation is None: # Then the image is a segmentation
        temp = (temp == __LA_LABEL).astype(temp.dtype)
    else:
        # Mask out anything outside of the LA
        segmentation = sitk.GetArrayFromImage(segmentation)
        temp = temp * (segmentation).astype(temp.dtype)
        
    selected_image = sitk.GetImageFromArray(temp)
    selected_image.CopyInformation(image)
        
    return selected_image
    

# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html    
def multiscale_demons(registration_algorithm: Callable, target_image: sitk.Image,
                      moving_image: sitk.Image, initial_transform : Optional[sitk.Transform] = None, 
                      shrink_factors: Optional[List[float]] = None,
                      smoothing_sigmas: Optional[List[float]] = None,
                      sampling: int = sitk.sitkLinear) -> sitk.DisplacementFieldTransform:
    """
    Run the given registration algorithm in a multiscale fashion.

    Parameters
    ----------
    registration_algorithm : Callable
        Any registration algorithm that has an Execute(target_image, moving_image,
                                                       displacement_field_image) method.
    target_image : sitk.Image
        Resulting transformation maps points from this image's spatial domain
        to the moving image spatial domain.
    moving_image : sitk.Image
        Resulting transformation maps points from the target_image's spatial
        domain to this image's spatial domain.
    initial_transform : sitk.Transform, optional
        Any SimpleITK transform, used to initialize the displacement field.
        The default is None.
    shrink_factors : List[float], optional
        Shrink factors relative to the original image's size. When the list entry,
        shrink_factors[i], is a scalar the same factor is applied to all axes.
        When the list entry is a list, shrink_factors[i][j] is applied to axis j.
        The default is None.
    smoothing_sigmas : List[float], optional
        Amount of smoothing which is done prior to resampling the image using
        the given shrink factor. These are in physical (image spacing) units.
        The default is None.
    sampling : int, optional
        The type of interpolation to be used when resampling the image. The
        default is sitk.sitkLinear.

    Returns
    -------
    displacement_field : sitk.DisplacementFieldTransform
        The resulting displacement field transform.

    """
    # Create image pyramid.
    target_images = [target_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            target_images.append(smooth_and_resample(target_images[0], shrink_factor, smoothing_sigma, sampling))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma, sampling))
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       target_images[-1].GetSize(),
                                                                       target_images[-1].GetOrigin(),
                                                                       target_images[-1].GetSpacing(),
                                                                       target_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(target_images[-1].GetWidth(), 
                                                target_images[-1].GetHeight(),
                                                target_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(target_images[-1])
 
    # Run the registration.            
    updated_displacement_field = registration_algorithm.Execute(target_images[-1], 
                                                                moving_images[-1], 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in reversed(list(zip(target_images[0:-1], moving_images[0:-1]))):
            updated_displacement_field = sitk.Resample(updated_displacement_field, f_image)
            updated_displacement_field = registration_algorithm.Execute(f_image, m_image,
                                                                        updated_displacement_field)
            
    return sitk.DisplacementFieldTransform(updated_displacement_field)


def nonrigid_register(target_image: sitk.Image, target_segmentation: sitk.Image,
                      moving_image: sitk.Image, moving_segmentation: sitk.Image,
                      initial_alignment: Optional[sitk.Transform] = None,
                      smooth: bool = True) -> sitk.DisplacementFieldTransform:
    """
    Perform non-rigid registration between the moving image and the target image
    using the Symmetric Forces Demons algorithm.

    Parameters
    ----------
    target_image : sitk.Image
        The target image to register to.
    target_segmentation : sitk.Image
        The target segmentation image.
    moving_image : sitk.Image
        The moving image to register.
    moving_segmentation : sitk.Image
        The moving segmentation image.
    initial_alignment : sitk.Transform, optional
        The initial alignment transform. Default is None.
    smooth : bool, optional
        Flag indicating whether to apply smoothing to the displacement field.
        Default is True.

    Returns
    -------
    displacement_field : sitk.DisplacementFieldTransform
        The resulting displacement field transform that transforms the moving
        image to the target image.

    """
    demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(1000)
    demons_filter.SetMaximumRMSError(0.00001)
    # Regularization
    if smooth:
        demons_filter.SetMaximumKernelWidth(60)
        demons_filter.SetSmoothDisplacementField(True)
    
    target_segmentation = sitk.Cast(target_segmentation, sitk.sitkFloat32)
    moving_segmentation = sitk.Cast(moving_segmentation, sitk.sitkFloat32)
                                                       
                                                       
    displacement_field = multiscale_demons(demons_filter, target_segmentation, moving_segmentation,
                                           initial_transform=initial_alignment,
                                           shrink_factors=[8, 4, 2, 1],
                                           smoothing_sigmas=[4, 2, 1, 0],
                                           sampling=sitk.sitkLinear)
    
    return displacement_field
    

def nonrigid_register_cases(moving_images: List[sitk.Image], moving_segmentations: List[sitk.Image],
                            target_image: sitk.Image, target_segmentation: sitk.Image,
                            affine_transformations: List[sitk.Transform]) -> List[sitk.DisplacementFieldTransform]:
    """
    Perform non-rigid registration for a list of moving images and segmentations
    to a target image and segmentation.

    Parameters
    ----------
    moving_images : List[sitk.Image]
        List of moving images to be registered.
    moving_segmentations : List[sitk.Image]
        List of moving segmentations corresponding to the moving images.
    target_image : sitk.Image
        Target image to which the moving images will be registered.
    target_segmentation : sitk.Image
        Target segmentation to which the moving segmentations will be registered.
    affine_transformations : List[sitk.Transform]
        List of initialization affine transformations of the moving images and
        segmentations.

    Returns
    -------
    displacement_fields : List[sitk.DisplacementFieldTransform]
        List of displacement field transformations representing the registration
        of moving images and segmentations to the target image and segmentation.

    """
    for i in range(len(moving_images)):
        moving_segmentations[i] = select_la(moving_segmentations[i], segmentation=None)
        moving_images[i] = select_la(moving_images[i], segmentation=moving_segmentations[i])
        
    target_segmentation = select_la(target_segmentation, segmentation=None)
    target_image = select_la(target_image, segmentation=target_segmentation)
    
    displacement_fields = []
    for i in range(len(transformed_images)):
        # Deform target image to each of the transformed
        displacement_field = nonrigid_register(target_image, target_segmentation,
                                               moving_images[i], moving_segmentations[i],
                                               initial_alignment=None)
        displacement_fields.append(displacement_field)          
        
    return displacement_fields


def transform_images(moving_images: List[sitk.Image], moving_segmentations: List[sitk.Image],
                     target_image: sitk.Image, target_segmentation: sitk.Image,
                     transformations: List[sitk.Transform], la_only: bool = False) -> \
                        Tuple[List[sitk.Image], List[sitk.Image]]:
    """
    Transforms a list of moving images and segmentations to align with a target
    image and segmentation using the provided transformations.
    
    Parameters
    ----------
    moving_images : List[sitk.Image]
        The list of moving images to be transformed.
    moving_segmentations : List[sitk.Image]
        The list of moving segmentations to be transformed.
    target_image : sitk.Image
        The target image to align the moving images with.
    target_segmentation : sitk.Image
        The target segmentation to align the moving segmentations with.
    transformations : List[sitk.Transform]
        The list of transformations to apply to the moving images and segmentations.
    la_only : bool, optional
        If True, only the left atrium (LA) region will be transformed.
        The default is False.

    Returns
    -------
    transformed_images : List[sitk.Image]
        The list of transformed moving images.
    transformed_segmentations : List[sitk.Image]
        The list of transformed moving segmentations.

    """
    if la_only:
        target_segmentation = select_la(target_segmentation, segmentation=None)
        target_image = select_la(target_image, segmentation=target_segmentation)
        
    transformed_images = []
    transformed_segmentations = []
    for i in range(len(moving_images)):
        if la_only:
            moving_segmentations[i] = select_la(moving_segmentations[i], segmentation=None)
            moving_images[i] = select_la(moving_images[i], segmentation=moving_segmentations[i])
            
        transformed_image = sitk.Resample(moving_images[i], target_image,
                                          transformations[i], sitk.sitkLinear, 0.0,
                                          target_image.GetPixelID())
        transformed_images.append(transformed_image)
        
        transformed_segmentation = sitk.Resample(moving_segmentations[i], target_segmentation,
                                                 transformations[i], sitk.sitkNearestNeighbor, 0,
                                                 target_segmentation.GetPixelID())
        transformed_segmentations.append(transformed_segmentation)
        
    return transformed_images, transformed_segmentations
        
    
def average_displacement_field(displacement_fields: List[sitk.DisplacementFieldTransform]) -> \
                sitk.DisplacementFieldTransform:
    """
    Calculates the average displacement field from a list of displacement fields.
    
    Parameters
    ----------
    displacement_fields : List[sitk.DisplacementFieldTransform]
        A list of displacement fields to calculate the average from.

    Returns
    -------
    mean_dvf : sitk.DisplacementFieldTransform
        The average displacement field.

    """
    dtype = sitk.GetArrayViewFromImage(displacement_fields[0].GetDisplacementField()).dtype
    size = list(displacement_fields[0].GetDisplacementField().GetSize())
    size.append(3)
    
    mean_dvf = np.zeros(size, dtype)
    
    for i in range(len(displacement_fields)):
        disp_field = displacement_fields[i].GetDisplacementField()
        dvf = sitk.GetArrayFromImage(disp_field)
        dvf = np.swapaxes(dvf, 0, 2)
        mean_dvf += dvf
        
    mean_dvf = mean_dvf / len(displacement_fields)
    
    mean_dvf = mean_dvf.astype(dtype)
    mean_dvf = np.swapaxes(mean_dvf, 0, 2)
    mean_dvf = sitk.GetImageFromArray(mean_dvf)
    mean_dvf.CopyInformation(displacement_fields[0].GetDisplacementField())
    mean_dvf = sitk.DisplacementFieldTransform(mean_dvf)
    
    return mean_dvf


def inverse_displacement_field(displacement_field: sitk.DisplacementFieldTransform) -> \
                sitk.DisplacementFieldTransform:
    """
    Calculates the inverse displacement field of a given displacement field.

    Parameters
    ----------
    displacement_field : sitk.DisplacementFieldTransform
        The input displacement field.

    Returns
    -------
    inverse_displacement : sitk.DisplacementFieldTransform
        The inverse displacement field.

    """
    displacement_field = displacement_field.GetDisplacementField()
    inverse_displacement = sitk.InvertDisplacementField(displacement_field,
                                                        maximumNumberOfIterations=40,
                                                        maxErrorToleranceThreshold=0.001,
                                                        meanErrorToleranceThreshold=0.00001,
                                                        enforceBoundaryCondition=True)
    inverse_displacement = sitk.DisplacementFieldTransform(inverse_displacement)
    
    return inverse_displacement
    

def transformation_to_deformation(transformation: sitk.Transform,
                                  reference_image: sitk.Image) -> sitk.DisplacementFieldTransform:
    """
    Converts a given transformation to a deformation field.

    Parameters
    ----------
    transformation : sitk.Transform
        The transformation to convert.
    reference_image : sitk.Image
        The reference image used to define the size and spacing of the
        deformation field.

    Returns
    -------
    deformation : sitk.DisplacementFieldTransform
        The resulting deformation field.

    """
    size = list(reference_image.GetSize())
    size.append(3)
    deformation = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                p = reference_image.TransformIndexToPhysicalPoint([i, j, k])
                displaced_point = transformation.TransformPoint(p)
                deformation[i, j, k, 0] = displaced_point[0] - p[0]
                deformation[i, j, k, 1] = displaced_point[1] - p[1]
                deformation[i, j, k, 2] = displaced_point[2] - p[2]
    
    deformation = np.swapaxes(deformation, axis1=0, axis2=2)
    deformation = sitk.GetImageFromArray(deformation, isVector=True)
    deformation.CopyInformation(reference_image)
    deformation = sitk.DisplacementFieldTransform(deformation)
    
    return deformation


def transformation_to_deformation_cases(moving_segmentations: List[sitk.Image],
                                        affine_transformations: List[sitk.Transform]) -> \
                    List[sitk.DisplacementFieldTransform]:
    """
    Converts a list of affine transformations to a list of deformation fields.

    Parameters
    ----------
    moving_segmentations : List[sitk.Image]
        The list of moving segmentations.
    affine_transformations : List[sitk.Transform]
        The list of affine transformations.

    Returns
    -------
    affine_displacements : List[sitk.DisplacementFieldTransform]
        The list of resulting affine displacement fields.

    """
    affine_displacements = []
    for i in range(len(affine_transformations)):
        affine_r = affine_transformations[i].GetNthTransform(1).GetNthTransform(0)
            
        # landmark transformation is only used to align
        # affine translation is used to capture the diversity
        affine_displacement = transformation_to_deformation(affine_r, moving_segmentations[i])
        affine_displacements.append(affine_displacement)

    return affine_displacements


def create_composite_transform(affine_transformations: List[sitk.Transform],
                               nonrigid_displacements: List[sitk.Transform],
                               inverse_average_affine: sitk.Transform,
                               inverse_average_nonrigid: sitk.Transform) -> List[sitk.Transform]:
    """
    Creates composite transformations by combining affine transformations,
    non-rigid displacements, and their inverse averages.

    Parameters
    ----------
    affine_transformations : List[sitk.Transform]
        The list of affine transformations.
    nonrigid_displacements : List[sitk.Transform]
        The list of nonrigid displacements.
    inverse_average_affine : sitk.Transform
        The inverse of the average affine transformation.
    inverse_average_nonrigid : sitk.Transform
        The inverse of the average non-rigid displacement.

    Returns
    -------
    composite_transformations : List[sitk.Transform]
        The list of composite transformations.

    """
    composite_transformations = []
    for i in range(len(moving_images)):
        t = sitk.CompositeTransform([affine_transformations[i], nonrigid_displacements[i],
                                     inverse_average_nonrigid, inverse_average_affine])
        composite_transformations.append(t)
        
    return composite_transformations


def average_segmentation(segmentation_list: List[sitk.Image], smooth: bool = False) -> sitk.Image:
    """
    Calculates the average segmentation from a list of segmentations using a
    majority vote.

    Parameters
    ----------
    segmentation_list : List[sitk.Image]
        A list of sitk.Image objects representing the segmentations.
    smooth : bool, optional
        Flag indicating whether to apply smoothing to the resulting segmentation.
        Default is False.

    Returns
    -------
    average_segmentation : sitk.Image
        The average segmentation obtained through majority voting.

    """
    if len(segmentation_list) <= 0:
        return None
    
    if len(segmentation_list) == 1:
        return segmentation_list[0]
    
    dtype = sitk.GetArrayViewFromImage(segmentation_list[0]).dtype
    sum_image = np.zeros(segmentation_list[0].GetSize(), dtype=dtype)
    for i in range(len(segmentation_list)):
        segmentation = sitk.GetArrayFromImage(segmentation_list[i])
        sum_image += segmentation
    
    # Majority vote
    threshold = np.floor(np.asarray([len(segmentation_list) / 2]))
    majority_image = (sum_image >= threshold).astype(sum_image.dtype)
    
    majority_image = sitk.GetImageFromArray(majority_image)
    majority_image.CopyInformation(segmentation_list[0])
    
    if smooth:
        # Smooth to remove noise
        majority_image = sitk.Median(majority_image, [3, 3, 3])
    
    return majority_image


def average_image(image_list: List[sitk.Image]) -> sitk.Image:
    """
    Calculates the average image from a list of images.

    Parameters
    ----------
    image_list : List[sitk.Image]
        A list of sitk.Image objects representing the images.

    Returns
    -------
    mean_image : sitk.Image
        The average image.

    """
    if len(image_list) <= 0:
        return None
    
    if len(image_list) == 1:
        return image_list[0]
    
    dtype = sitk.GetArrayViewFromImage(image_list[0]).dtype
    sum_image = np.zeros(image_list[0].GetSize(), dtype=dtype)
    for i in range(len(image_list)):
        image = sitk.GetArrayFromImage(image_list[i])
        sum_image += image
        
    mean_image = sum_image / len(image_list)
    mean_image = sitk.GetImageFromArray(mean_image)
    mean_image.CopyInformation(image_list[0])
    
    return mean_image
    

def generate_mesh(segmentation: np.ndarray) -> pv.PolyData:
    """
    Generates a mesh from a 3D segmentation using the marching cubes algorithm.

    Parameters
    ----------
    segmentation : np.ndarray
        A 3D numpy array representing the segmentation.

    Returns
    -------
    mesh : pv.PolyData
        The generated mesh as a pv.PolyData object.

    """
    verts, faces, normals, values = measure.marching_cubes(segmentation, level=0.5,
                                                           gradient_direction='descent',
                                                           allow_degenerate=False,
                                                           method='lewiner')
    vfaces = np.column_stack((np.ones(len(faces),) * 3, faces)).astype(int)
    
    mesh = pv.PolyData(verts, vfaces)
    mesh['Normals'] = normals
    mesh['values'] = values
    mesh = mesh.triangulate().clean()
    
    return mesh


def smooth_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """
    Smooths and refines a mesh using various operations.

    Parameters
    ----------
    mesh : pv.PolyData
        The input mesh to be smoothed and refined.

    Returns
    -------
    mesh : pv.PolyData
        The smoothed and refined mesh.

    """
    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.01)
    mesh = mesh.subdivide(1, 'linear')
    mesh = mesh.smooth(n_iter=10, feature_smoothing=False)
    mesh = mesh.clean().triangulate()
    
    return mesh

    
if __name__ == '__main__':
    print('Executing atlas construction...')
    base_path = os.path.join('..', '..', 'data_nn', 'train')   
    patients = filter_cases(sorted(os.listdir(base_path)))
    
    ###########
    ## Load data
    ###########
    print('Loading data...')
    reference_image, reference_segmentation = load_reference_data(base_path, patients)
    # Load all patient data
    moving_images, moving_segmentations = load_patient_data(base_path, patients,
                                                            reference_image, reference_segmentation)
    
    ###########
    ## Register cases to reference case
    ###########
    # Initial alignment and affine registration
    print('Performing affine step...')
    affine_transformations = affine_register_cases(moving_images, moving_segmentations,
                                                   reference_image, reference_segmentation)
    
    # Nonrigid registration
    print('Performing nonrigid step...')
    transformed_images, transformed_segmentations = transform_images(
        moving_images, moving_segmentations, reference_image, reference_segmentation,
        affine_transformations)
    nonrigid_displacements = nonrigid_register_cases(transformed_images, transformed_segmentations,
                                                     reference_image, reference_segmentation,
                                                     affine_transformations)

    ###########
    ## Obtain the inverse average transformations
    ###########
    print('Combining forward and inverse steps...')
    # Convert the affine transformations to displacement fields
    affine_displacements = transformation_to_deformation_cases(moving_segmentations, affine_transformations)
    inverse_average_affine = inverse_displacement_field(average_displacement_field(affine_displacements))
    inverse_average_nonrigid = inverse_displacement_field(average_displacement_field(nonrigid_displacements))
    # Combine the forward and inverse steps
    composite_transformations = create_composite_transform(affine_transformations,
                                                           nonrigid_displacements,
                                                           inverse_average_affine,
                                                           inverse_average_nonrigid)
    
    ###########
    ## Construct atlas
    ###########
    print('Constructing atlas...')
    # Transform cases to atlas space
    transformed_images, transformed_segmentations = transform_images(
        transformed_images, transformed_segmentations, reference_image, reference_segmentation,
        composite_transformations, la_only=True)   
    
    atlas_image = average_image(transformed_images)
    atlas_segmentation = average_segmentation(transformed_segmentations, smooth=True)
    # Obtain the mesh from the atlas
    atlas_segmentation_n = sitk.GetArrayFromImage(atlas_segmentation)
    atlas_segmentation_n = np.swapaxes(atlas_segmentation_n, axis1=0, axis2=2)
    atlas_mesh = generate_mesh(atlas_segmentation_n)
    atlas_mesh = smooth_mesh(atlas_mesh)
    
    print('Saving atlas...')
    # Save the generated atlas
    output_folder = '_atlas_output'
    os.makedirs(output_folder, exist_ok=True)
    sitk.WriteImage(atlas_image, os.path.join(output_folder, 'atlas_image.nii.gz'))
    sitk.WriteImage(atlas_segmentation, os.path.join(output_folder, 'atlas_segmentation.nii.gz'))
    atlas_mesh.save(os.path.join(output_folder, 'atlas_mesh.vtk'))
    
    # Save the reference image used
    sitk.WriteImage(reference_image, os.path.join(output_folder, 'reference_image.nii.gz'))
    sitk.WriteImage(reference_segmentation, os.path.join(output_folder, 'reference_segmentation.nii.gz'))
