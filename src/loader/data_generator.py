import os
import gc

import random
from enum import Enum

from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
from glob import glob

import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage import morphology

import SimpleITK as sitk


class DatasetType(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'
        
        
        
class BaseDataLoader():
    
    def __init__(self, image_size: Tuple[int] = (96, 96, 36), memory_cache: bool = True,
                 disk_cache: bool = True, dilation_radius: Optional[float] = None,
                 patient_case: Optional[str] = None, translation_alignment: bool = False,
                 data_type: Optional[str] = None) -> None:
        """
        Initializes the base data loader.

        Parameters
        ----------
        image_size : Tuple[int], optional
            The target crop size of the image. The default is (96, 96, 36).
        memory_cache : bool, optional
            Whether to save the data in memory. The default is True.
        disk_cache : bool, optional
            Whether to save the data to disk. The default is True.
        dilation_radius : float, optional
            The dilation radius to apply to the contour. If None, then then contour
            is not calculated. The default is None.
        patient_case : str, optional
            Which patient cases to include in the data generator. If None, then
            all available case will be loaded. The default is None.
        translation_alignment : bool, optional
            Whether to align the phases of a case. The default is False.
        data_type : str, optional
            From which data folder to process the data. Default is None.

        Returns
        -------
        None

        """
        self.disk_cache = disk_cache
        self.memory_cache = memory_cache
        self.data_in_memory = {}
        
        self.dilation_radius = dilation_radius
        self.translation_alignment = translation_alignment
        
        self.data_directory, self.cache_directory = BaseDataLoader.get_directories(data_type,
                                                                                   dilation_radius)
        
        self.train_directory = Path(os.path.join(self.data_directory, 'train'))
        # Fallback to cached directories if main data does not exist in directory
        if not os.path.isdir(self.train_directory):
            self.train_directory = Path(os.path.join(self.cache_directory, 'train'))
        self.train_list = self.get_patient_list(self.train_directory, patient_case)
        
        self.image_size = image_size + (1,)
        self.flow_size = image_size + (3,)
        self.phase_embedding_size = (20,)
        self.num_classes = 2
        
        self.list_shuffle = random.Random(0)
    
        
    @staticmethod
    def get_directories(data_type: Optional[str] = None,
                        dilation_radius: Optional[float] = None) -> Tuple[Path]:
        """
        Returns the expected data and cached data folders.

        Parameters
        ----------
        data_type : str, optional
            The type of data that will be used. The default is None.
        dilation_radius : float, optional
            The amount of dilation to apply to the contour. This defines the
            naming scheme for the cached folder. The default is None.

        Returns
        -------
        data_directory : Path
            The path to the expected location of the data folder.
        cache_directory : Path
            The path to the location of the cached data folder.

        """
        file_path = Path(__file__).parent.absolute()
        data_folder = 'data'
        if data_type:
            data_folder += ('_' + data_type)
        expected_data_directory = os.path.join('..', '..', data_folder)
        data_directory = Path(os.path.join(file_path, expected_data_directory))
        
        cache_directory = os.path.join('..', '..', data_folder + '_cache')
        if dilation_radius is not None:
            name = 'contour_3d_' + str(dilation_radius)
            cache_directory = os.path.join('..', '..', data_folder + '_' + name + '_cache')
        cache_directory = Path(os.path.join(file_path, cache_directory))
        
        return data_directory, cache_directory
    
    
    @staticmethod
    def get_patient_list(root_directory: Union[str, Path], patient_case: Optional[str] = None) -> List[Path]:
        """
        Returns a list of all patient paths. If patient_case is given, then
        a list with that specific case is returned.

        Parameters
        ----------
        root_directory : Union[str, Path]
            The root directory in which the patient folders exist.
        patient_case : str, optional
            Specific patient case to return. The default is None.

        Returns
        -------
        files : List[Path]
            A list of paths to patient folders.

        """
        files = glob(os.path.join(root_directory, "**"))
        if patient_case: # Select only one case
            files_selected = []
            for file in files:
                folder = Path(file).parts[-1]
                if folder.lower() == patient_case.lower():
                    files_selected.append(file)
                    
            files = files_selected
        files = [Path(i) for i in files]
        
        return files
        
        
    @staticmethod
    def load_image(file_path: Union[str, Path]) -> sitk.Image:
        """
        Reads the image as a SimpleITK image.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path to the image.

        Returns
        -------
        image : SimpleITK.Image
            The read image.

        """
        if not os.path.exists(file_path):
            return None
        image = sitk.ReadImage(file_path)
    
        return image
    
    
    @staticmethod
    def load_patient_data(patient_directory: Union[str, Path]) -> Dict[str, List[sitk.Image]]:
        """
        Loads the patient image and segmentation sequence in a list. The lists
        are returned as a dictionary with the keys 'images' and 'segmentations'
        respectively. If the displacements exists, then return an additional
        key 'displacements'.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The path to the patient folder.

        Returns
        -------
        patient_data : Dict[str, List[SimpleITK.Image]]
            Returns the patient images and segmentations (and displacements, if
            they exist) accross the whole sequence as a dictionary.

        """
        patient_data = {}
        image_list = []
        segmentation_list = []
        displacement_list = []
        
        image_path = os.path.join(patient_directory, 'images')
        segmentation_path = os.path.join(patient_directory, 'segmentations')
        displacement_path = os.path.join(patient_directory, 'displacements')
        
        for file in sorted(os.listdir(image_path)):
            file_path = os.path.join(image_path, file)
            image = BaseDataLoader.load_image(file_path)
            
            file_path = os.path.join(segmentation_path, file)
            segmentation = BaseDataLoader.load_image(file_path)
            segmentation = segmentation == 1 # Only select Left Atrium labels
            
            file_path = os.path.join(displacement_path, file)
            displacement = BaseDataLoader.load_image(file_path)
            if not displacement is None:
                displacement_list.append(displacement)
            
            image_list.append(image)
            segmentation_list.append(segmentation)
            
        
        patient_data['images'] = image_list
        patient_data['segmentations'] = segmentation_list
        if len(displacement_list) > 0:
            patient_data['displacements'] = segmentation_list
        
        return patient_data
    
    
    @staticmethod
    def cached_file_name(time_index: int) -> str:
        """
        Obtains the expected cached file name based on the time sequence index.

        Parameters
        ----------
        time_index : int
            The time index to return the expected cache name for.

        Returns
        -------
        cached_file_name : str
            The cached file name.

        """
        return '{:02d}.nii.gz'.format(time_index)     

    
    def get_cache_directory(self, patient_directory: Union[str, Path]) -> Path:
        """
        Returns the cache directory for the given patient directory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the patient folder.

        Returns
        -------
        cache_directory : Path
            Returns a path to the expected location of the cache folder of the
            given patient.

        """
        path = os.path.normpath(patient_directory)
        split_path = path.split(os.sep)
        # .. / data / training, validation or testing / patient ID
        # only last two are of interest
        cache_directory = Path(os.path.join(self.cache_directory,
                                            split_path[-2],
                                            split_path[-1]))
        
        return cache_directory
    
    
    def is_cached(self, patient_directory: Union[str, Path]) -> bool:
        """
        Checks if the given patient directory is cached. To be considered as
        cached, the cached folder and non-empty subfolders ('images' and
        'segmentations') exist.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The patient directory to check if it has a respective cache directory.

        Returns
        -------
        cached : bool
            Returns True if the cached directory exists.

        """
        patient_cache_directory = self.get_cache_directory(patient_directory)
        
        # Check if folder exists
        if not os.path.isdir(patient_cache_directory):
            return False
        
        subfolders = ['images', 'segmentations']
        
        # Check if every individual file exists
        for image_type in subfolders:
            folder_path = os.path.join(patient_cache_directory, image_type)
            
            if not os.path.exists(folder_path):
                return False
                              
            file_list = os.listdir(folder_path)
            if len(file_list) == 0:
                return False
            

        return True
    
    
    def save_cache(self, patient_directory: Union[str, Path],
                    patient_data: Dict[str, List[sitk.Image]]) -> None:
        """
        Saves the patient data to cache, if self.disk_cache is set to True.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the non-cached patient data.
        patient_data : Dict[str, List[SimpleITK.Image]]
            A dictionary containing a list of 'images' and 'segmentations'.

        Returns
        -------
        None

        """
        if not self.disk_cache:
            return
        
        patient_cache_directory = self.get_cache_directory(patient_directory)
        os.makedirs(patient_cache_directory, exist_ok=True)
        
        for key, data in patient_data.items():
            folder_path = os.path.join(patient_cache_directory, key)
            os.makedirs(folder_path, exist_ok=True)
            
            for i in range(len(data)):
                file_path = os.path.join(folder_path, self.cached_file_name(i))
                sitk.WriteImage(data[i], file_path)
                
    
    def load_cache(self, patient_directory: Union[str, Path]) -> Dict[str, List[sitk.Image]]:
        """
        Loads the data from the cache. Assumes cache directory exists and is
        populated correctly.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the non-cached patient data.

        Returns
        -------
        patient_data : Dict[str, List[SimpleITK.Image]]
            Returns a dictionary containing a list of 'images' and 'segmentations'
            from the cached directory.

        """
        patient_cache_directory = self.get_cache_directory(patient_directory)
        patient_data = self.load_patient_data(patient_cache_directory)
        
        return patient_data
    
    
    def is_in_memory(self, patient_directory: Union[str, Path]) -> bool:
        """
        Checks if the patient data is available in main memory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the patient data.

        Returns
        -------
        in_memory : bool
            Returns True if the data is in memory.

        """
        if patient_directory in self.data_in_memory:
            return True

        return False
    
    
    def save_memory(self, patient_directory: Union[str, Path],
                    patient_data: Dict[str, List[sitk.Image]]) -> None:
        """
        Saves the data into memory (dictionary).

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the patient data.
        patient_data : Dict[str, List[sitk.Image]]
            Patient data as a dictionary containing a list of 'images' and
            'segmentations'.

        Returns
        -------
        None

        """
        if self.memory_cache:
            self.data_in_memory[patient_directory] = patient_data.copy()
            
            
    def get_memory(self, patient_directory: Union[str, Path]) -> Dict[str, List[sitk.Image]]:
        """
        Returns a shallow copy of the patient data from memory.

        Parameters
        ----------
        patient_directory : Union[str, Path]
            The directory to the patient data.

        Returns
        -------
        patient_data : Dict[str, List[SimpleITK.Image]]
            Returns a dictionary containing a list of 'images' and 'segmentations'
            from the cached directory.

        """
        patient_data = self.data_in_memory[patient_directory]
        return patient_data.copy()
    
    
    @staticmethod
    def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
        """
        Resamples an image to given element spacing and output size.

        Parameters
        ----------
        image : sitk.Image
            The image to be resampled.
        out_spacing : Tuple[float], optional
            The desired element spacing for the resampled image. The default is (1.0, 1.0, 1.0).
        out_size : Tuple[float], optional
            The desired output size for the resampled image. The default is None.
        is_label : bool, optional
            Specifies whether the image is a label image. If True, nearest neighbor interpolation is used.
            If False, linear interpolation is used. The default is False.
        pad_value : int, optional
            The default pixel value for areas outside the original image bounds. The default is 0.

        Returns
        -------
        resampled_image : sitk.Image
            The resampled image.

        """
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)
    
        return resample.Execute(image)
        
    
    @staticmethod
    def find_edge(segmentation_sitk: sitk.Image, dilation_radius: float = 0) -> sitk.Image:
        """
        Finds the edge of a segmentation image.

        Parameters
        ----------
        segmentation_sitk : sitk.Image
            The input binary segmentation image.
        dilation_radius : float, optional
            The radius for dilating the edge. Default is 0.

        Returns
        -------
        edge : sitk.Image
            The edge image.

        """
        segmentation = sitk.GetArrayFromImage(segmentation_sitk).astype(int)
        
        # Get segmentation boundaries
        edge = find_boundaries(segmentation, connectivity=2, mode='inner', background=0).astype(int)
        
        # Dilate edge for more flexibility when computing the vector field
        if dilation_radius > 0:
            structure = morphology.ball(radius=dilation_radius)
            edge = ndimage.binary_dilation(edge, structure, iterations=1).astype(int)
            edge *= segmentation
        
        edge = sitk.GetImageFromArray(edge)
        edge.CopyInformation(segmentation_sitk)
        
        return edge        
        
    
    @staticmethod 
    def preprocess_data(data: Dict[str, List[sitk.Image]],
                        dilation_radius: Optional[float] = None) -> Dict[str, List[sitk.Image]]:
        """
        Preprocesses the data by extracting the contour.

        Parameters
        ----------
        data : Dict[str, List[sitk.Image]]
            The input data dictionary containing images and segmentations.
        dilation_radius : float, optional
            The dilation radius of the contour. The default is None.

        Returns
        -------
        Dict[str, List[sitk.Image]]
            The preprocessed data dictionary.

        """
        if dilation_radius is not None and dilation_radius >= 0:
            for i in range(len(data['segmentations'])):    
                contour = BaseDataLoader.find_edge(data['segmentations'][i],
                                                   dilation_radius=dilation_radius)
                data['segmentations'][i] = contour
        
        return data


    @staticmethod
    def shift_array(array, offset, constant_values=0):
        """
        Returns a copy of the array shifted by the specified offset.
        (https://stackoverflow.com/a/70297929)

        Parameters:
        ----------
        array : np.ndarray
            The input array to be shifted.
        offset : int or tuple of ints
            The offset by which the array should be shifted along each axis.
        constant_values : int, optional
            The value to be used for filling the shifted regions of the array. Default is 0.

        Returns:
        ----------
        shifted_array : np.ndarray
            The shifted array.

        """
        array = np.asarray(array)
        offset = np.atleast_1d(offset)
        assert len(offset) == array.ndim
        new_array = np.empty_like(array)
        
        def slice1(o):
          return slice(o, None) if o >= 0 else slice(0, o)
        
        new_array[tuple(slice1(o) for o in offset)] = (
            array[tuple(slice1(-o) for o in offset)])
        
        for axis, o in enumerate(offset):
          new_array[(slice(None),) * axis +
                    (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values
        
        return new_array
    
    
    @staticmethod
    def align_data(source_image: np.ndarray, source_segmentation: np.ndarray,
                   target_image: np.ndarray, target_segmentation: np.ndarray) -> Tuple[np.ndarray]:
        """
        Aligns the source image and segmentation to the target based on their segmentation's centers of mass.

        Parameters:
        ----------
        source_image : np.ndarray
            The source image to be aligned.
        source_segmentation : np.ndarray
            The source segmentation to be aligned.
        target_image : np.ndarray
            The target image to align to.
        target_segmentation : np.ndarray
            The target segmentation to align to.

        Returns:
        ----------
        Tuple[np.ndarray]:
            A tuple containing the aligned source image and segmentation.

        """
        source_centre = np.asarray(ndimage.center_of_mass(source_segmentation >= 1))
        target_centre = np.asarray(ndimage.center_of_mass(target_segmentation >= 1))
        
        shift_centre = np.rint(target_centre - source_centre).astype(int)
        
        shifted_source_image = BaseDataLoader.shift_array(source_image, shift_centre, constant_values=0)
        shifted_source_segmentation = BaseDataLoader.shift_array(source_segmentation, shift_centre, constant_values=0)
        
        return shifted_source_image, shifted_source_segmentation
        
        
    @staticmethod
    def to_numpy(image: sitk.Image, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts a SimpleITK image to a NumPy array.

        Parameters:
        ----------
        image : sitk.Image
            The SimpleITK image to be converted.
        dtype : np.dtype, optional
            The desired data type of the resulting NumPy array. Default is np.float32.

        Returns:
        ----------
        numpy_image : np.ndarray:
            The converted NumPy array.

        """
        numpy_image = sitk.GetArrayFromImage(image)
        # Swap axes so ordering is x, y, z rather than z, y, x as stored
        # in sitk
        numpy_image = np.swapaxes(numpy_image, 0, -1)
        numpy_image = numpy_image.astype(dtype)
        
        return numpy_image
    
    
    def generator(self, patient_directory: Union[str, Path]) -> Tuple[Dict[str, List[np.ndarray]]]:
        """
        Generates patient data for a given patient directory.

        If the patient data is already in memory, it retrieves it from memory.
        If the patient data is cached, it loads it from the cache, saves it to memory, and returns it.
        If the patient data is not cached, it loads the patient data, preprocesses it, saves it to cache and memory, and returns it.

        Parameters:
        ----------
        patient_directory : Union[str, Path]
            The directory path of the patient.

        Returns:
        ----------
        patient_data : Tuple[Dict[str, List[np.ndarray]]]
            The generated patient data, represented as a dictionary of lists of
            NumPy arrays. The first index of the tuple represents the model
            input and the second input the expected output of the model.

        """
        if self.is_in_memory(patient_directory):
            patient_data = self.get_memory(patient_directory)
        elif self.is_cached(patient_directory):
            patient_data = self.load_cache(patient_directory)
            self.save_memory(patient_directory, patient_data)
        else:
            patient_data = self.load_patient_data(patient_directory)
            patient_data = self.preprocess_data(patient_data,
                                                self.dilation_radius)
            self.save_cache(patient_directory, patient_data)
            self.save_memory(patient_directory, patient_data)

        return patient_data
    

    @staticmethod
    def to_structure(moving_image: sitk.Image, fixed_image: sitk.Image,
                     moving_segmentation: sitk.Image, fixed_segmentation: sitk.Image,
                     add_batch_axis: bool = False, align: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Converts the input images and segmentations into a structured data
        format based on the specified options. The returned tuple consists of
        the input and expected output data, respectively, of the network.

        Parameters:
        ----------
        moving_image : sitk.Image
            The moving image.
        fixed_image : sitk.Image
            The fixed image.
        moving_segmentation : sitk.Image
            The moving segmentation.
        fixed_segmentation : sitk.Image
            The fixed segmentation.
        add_batch_axis : bool, optional
            Whether to add a batch axis to the data. Defaults to False.
        align : bool, optional
            Whether to align the data. Defaults to False.

        Returns:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            The structured data, represented as a tuple of dictionaries containing
            NumPy arrays.

        """
        moving_image = BaseDataLoader.to_numpy(moving_image)
        fixed_image = BaseDataLoader.to_numpy(fixed_image)
        moving_segmentation = BaseDataLoader.to_numpy(moving_segmentation)
        fixed_segmentation = BaseDataLoader.to_numpy(fixed_segmentation)
        
        if align:
            fixed_image, fixed_segmentation = BaseDataLoader.align_data(fixed_image,
                                                                        fixed_segmentation,
                                                                        moving_image,
                                                                        moving_segmentation)
        
        if add_batch_axis:
            moving_image = np.expand_dims(moving_image, axis=0) 
            fixed_image = np.expand_dims(fixed_image, axis=0) 
            moving_segmentation = np.expand_dims(moving_segmentation, axis=0) 
            fixed_segmentation = np.expand_dims(fixed_segmentation, axis=0)
        
        flow_shape = fixed_image.shape + (3,)
        
        if moving_image.ndim == 3:
           moving_image = np.expand_dims(moving_image, axis=-1) 
        if fixed_image.ndim == 3:
           fixed_image = np.expand_dims(fixed_image, axis=-1) 
        if moving_segmentation.ndim == 3:
           moving_segmentation = np.expand_dims(moving_segmentation, axis=-1) 
        if fixed_segmentation.ndim == 3:
           fixed_segmentation = np.expand_dims(fixed_segmentation, axis=-1) 
        
        flow = np.zeros(flow_shape, dtype=np.float32)
        moving_image = moving_image * moving_segmentation
        fixed_image = fixed_image * fixed_segmentation
        
        data = ({'input_moving': moving_image,
                 'input_fixed': fixed_image,
                 'input_moving_seg': moving_segmentation},
                {'output_fixed': fixed_image,
                 'output_fixed_seg': fixed_segmentation,
                 'output_moving': moving_image,
                 'flow_params': flow,
                 'output_flow': flow})
            
        return data
        

    @staticmethod
    def clear_data(data: Tuple[Dict[str, np.ndarray]]) -> None:
        """
        Clears the data by deleting the arrays in the given data dictionary.

        Parameters:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            The data dictionary containing arrays to be cleared.

        Returns:
        ----------
        None

        """
        for i in range(len(data)):
            for key, array in data[i].items():
                del array
                
        del data
        

    def data_generator(self, patient_directory: Union[Path, str],
                       verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a given patient directory.

        Parameters:
        ----------
        patient_directory : Union[Path, str]
            The directory path of the patient data.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        if verbose > 0:
            print('Generating patient: ', patient_directory)
        patient_data = self.generator(patient_directory)
        
        j = 0
        for i in range(len(patient_data['images'])):  
            moving_image = patient_data['images'][j]
            fixed_image = patient_data['images'][i]
            moving_segmentation = patient_data['segmentations'][j]
            fixed_segmentation = patient_data['segmentations'][i]
            
            data = self.to_structure(moving_image, fixed_image, 
                                     moving_segmentation, fixed_segmentation,
                                     align=self.translation_alignment)
            yield data
            self.clear_data(data)
            
        gc.collect()
    
    
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a specific index in the dataset.

        Parameters:
        ----------
        index : int
            The index of the patient data in the dataset.
        dataset : Union[DatasetType, int]
            The type of dataset to generate data from.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        if dataset == DatasetType.train or dataset == 0:
            patient_directory = self.train_list[index]
            
        yield from self.data_generator(patient_directory, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, shuffle: bool = False,
                          verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a specific dataset type.

        Parameters:
        ----------
        dataset : DatasetType
            The type of dataset to generate data from.
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to False.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        if dataset == DatasetType.train:
            data_list = self.train_list
            
        if shuffle:
            self.list_shuffle.shuffle(data_list)
            
        for patient_directory in data_list:
            yield from self.data_generator(patient_directory, verbose)
                
                
    def train_generator(self, shuffle: bool = True, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates training data.

        Parameters:
        ----------
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to True.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        yield from self.dataset_generator(DatasetType.train, shuffle=shuffle,
                                          verbose=verbose)



class VoxelmorphDataLoader(BaseDataLoader):
    
    @staticmethod
    def to_structure(moving_image: sitk.Image, fixed_image: sitk.Image,
                     moving_segmentation: sitk.Image, fixed_segmentation: sitk.Image,
                     add_batch_axis: bool = False, align: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Converts the input images and segmentations into a structured data format
        for the Voxelmorph overlay data loader.
        
        Parameters:
        ----------
        moving_image : sitk.Image
            The moving image.
        fixed_image : sitk.Image
            The fixed image.
        moving_segmentation : sitk.Image
            The moving segmentation.
        fixed_segmentation : sitk.Image
            The fixed segmentation.
        add_batch_axis : bool, optional
            Whether to add a batch axis to the data. Defaults to False.
        align : bool, optional
            Whether to align the images and segmentations. Defaults to False.

        Returns:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.
        
        """
        moving_image = BaseDataLoader.to_numpy(moving_image)
        fixed_image = BaseDataLoader.to_numpy(fixed_image)
        moving_segmentation = BaseDataLoader.to_numpy(moving_segmentation)
        fixed_segmentation = BaseDataLoader.to_numpy(fixed_segmentation)
        
        if align:
            fixed_image, fixed_segmentation = BaseDataLoader.align_data(fixed_image,
                                                                        fixed_segmentation,
                                                                        moving_image,
                                                                        moving_segmentation)
        
        if add_batch_axis:
            moving_image = np.expand_dims(moving_image, axis=0) 
            fixed_image = np.expand_dims(fixed_image, axis=0) 
            moving_segmentation = np.expand_dims(moving_segmentation, axis=0) 
            fixed_segmentation = np.expand_dims(fixed_segmentation, axis=0)
        
        flow_shape = fixed_image.shape + (3,)
        
        if moving_image.ndim == 3:
           moving_image = np.expand_dims(moving_image, axis=-1) 
        if fixed_image.ndim == 3:
           fixed_image = np.expand_dims(fixed_image, axis=-1) 
        if moving_segmentation.ndim == 3:
           moving_segmentation = np.expand_dims(moving_segmentation, axis=-1) 
        if fixed_segmentation.ndim == 3:
           fixed_segmentation = np.expand_dims(fixed_segmentation, axis=-1) 
        
        flow = np.zeros(flow_shape, dtype=np.float32)
        masked_moving_image = moving_image * moving_segmentation
        masked_fixed_image = fixed_image * fixed_segmentation
        
        data = ({'vxm_source_input': masked_moving_image,
                 'vxm_target_input': masked_fixed_image},
                {'vxm_transformer': masked_fixed_image,
                 'vxm_flow': flow})
            
        return data
    
    
    
class VoxelmorphSegDataLoader(BaseDataLoader):
    
    @staticmethod
    def to_structure(moving_image: sitk.Image, fixed_image: sitk.Image,
                     moving_segmentation: sitk.Image, fixed_segmentation: sitk.Image,
                     add_batch_axis: bool = False, align: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Converts the input images and segmentations into a structured data format for the Voxelmorph segmentation data loader.
        
        Parameters:
        ----------
        moving_image : sitk.Image
            The moving image.
        fixed_image : sitk.Image
            The fixed image.
        moving_segmentation : sitk.Image
            The moving segmentation.
        fixed_segmentation : sitk.Image
            The fixed segmentation.
        add_batch_axis : bool, optional
            Whether to add a batch axis to the data. Defaults to False.
        align : bool, optional
            Whether to align the images and segmentations. Defaults to False.

        Returns:
        ----------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.
        
        """
        moving_image = BaseDataLoader.to_numpy(moving_image)
        fixed_image = BaseDataLoader.to_numpy(fixed_image)
        moving_segmentation = BaseDataLoader.to_numpy(moving_segmentation)
        fixed_segmentation = BaseDataLoader.to_numpy(fixed_segmentation)
        
        if align:
            fixed_image, fixed_segmentation = BaseDataLoader.align_data(fixed_image,
                                                                        fixed_segmentation,
                                                                        moving_image,
                                                                        moving_segmentation)
        
        if add_batch_axis:
            moving_image = np.expand_dims(moving_image, axis=0) 
            fixed_image = np.expand_dims(fixed_image, axis=0) 
            moving_segmentation = np.expand_dims(moving_segmentation, axis=0) 
            fixed_segmentation = np.expand_dims(fixed_segmentation, axis=0)
        
        flow_shape = fixed_image.shape + (3,)
        
        if moving_image.ndim == 3:
           moving_image = np.expand_dims(moving_image, axis=-1) 
        if fixed_image.ndim == 3:
           fixed_image = np.expand_dims(fixed_image, axis=-1) 
        if moving_segmentation.ndim == 3:
           moving_segmentation = np.expand_dims(moving_segmentation, axis=-1) 
        if fixed_segmentation.ndim == 3:
           fixed_segmentation = np.expand_dims(fixed_segmentation, axis=-1) 
        
        flow = np.zeros(flow_shape, dtype=np.float32)
        
        data = ({'vxm_dense_source_input': moving_image,
                 'vxm_dense_target_input': fixed_image,
                 'vxm_source_seg': moving_segmentation},
                {'vxm_dense_transformer': fixed_image,
                 'vxm_seg_transformer': fixed_segmentation,
                 'vxm_dense_flow': flow})
            
        return data