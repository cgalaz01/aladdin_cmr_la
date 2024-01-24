import numpy as np

import SimpleITK as sitk



def get_file_extension(output_type: str):
    """
    Returns the file type (int) and output extension.
    
    Parameters
    ----------
    output_type : str
        The output type to obtain the extension for.

    Returns
    -------
    output_type : int
        O for numpy and 1 for nifti.
    file_extension : str
        'npy' for numpy and 'nii.gz' for nifti.

    """
    if output_type.lower() == 'numpy':
        output_type = 0
        file_extension = 'npy'
    elif output_type.lower() == 'nifti':
        output_type = 1
        file_extension = 'nii.gz'
        
    return output_type, file_extension


def numpy_to_sitk(array: np.ndarray, sitk_ref_image: sitk.Image,
                  is_displacement_field: bool = False) -> sitk.Image:
    """
    Transforms a 3d numpy array to a SimpleITK image.

    Parameters
    ----------
    array : np.ndarray
        The 3D numpy array to transform.
    sitk_ref_image : sitk.Image
        A reference SimpleITK image to copy the header of.
    is_displacement_field : bool, optional
        Whether the array represents a displacement vector field. The default
        is False.

    Returns
    -------
    image : sitk.Image
        The SimpleITK image representing the give numpy array.

    """
    if is_displacement_field:
        displacement = np.swapaxes(array, 0, 2)
        displacement = displacement.astype(np.float64)
        displacement = sitk.GetImageFromArray(displacement)
        displacement.SetOrigin(sitk_ref_image.GetOrigin())
        displacement.SetSpacing(sitk_ref_image.GetSpacing())
        displacement.SetDirection(sitk_ref_image.GetDirection())
        displacement = sitk.DisplacementFieldTransform(displacement)
        image = displacement.GetDisplacementField()
        
    else:
        image = np.swapaxes(array, 0, -1)
        image = sitk.GetImageFromArray(image)
        image.CopyInformation(sitk_ref_image)
        
    return image


def save_numpy_to_sitk(image: np.ndarray, reference: sitk.Image, file_path: str,
                       is_displacement_field: bool = False) -> None:
    """
    Transforms the numpy to a SimpleITk image and saves it to file.

    Parameters
    ----------
    image : np.ndarray
        The 3D numpy array to save.
    reference : sitk.Image
        A reference SimpleITK image to copy the header of.
    file_path : str
        The output file path to save the image. Assumes directory already exists.
    is_displacement_field : bool, optional
        Whether the array represents a displacement vector field. The default
        is False.

    Returns
    -------
    None

    """
    sitk_image = numpy_to_sitk(image, reference,
                               is_displacement_field=is_displacement_field)
    sitk.WriteImage(sitk_image, file_path)