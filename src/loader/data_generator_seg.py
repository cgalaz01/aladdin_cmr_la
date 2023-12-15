import os
import gc

import random

from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
from glob import glob

import numpy as np


import SimpleITK as sitk

from loader.augment import DataAugmentation
        

        
class SegDataLoader():     
    
    def __init__(self, image_size: Tuple[int] = (96, 96, 36), memory_cache: bool = True,
                 disk_cache: bool = True, patient_case: Optional[str] = None) -> None:
        self.disk_cache = disk_cache
        self.memory_cache = memory_cache
        self.data_in_memory = {}
        
        file_path = Path(__file__).parent.absolute()
        expected_data_directory = os.path.join('..', '..', 'data')
        self.data_directory = Path(os.path.join(file_path, expected_data_directory))
        
        self.cache_directory = os.path.join('..', '..', 'data_cache')
        self.cache_directory = Path(os.path.join(file_path, self.cache_directory))
        
        self.train_directory = Path(os.path.join(self.data_directory, 'train'))
        # Fallback to cached directories if main data do not exist in directory
        if not os.path.isdir(self.train_directory):
            self.train_directory = Path(os.path.join(self.cache_directory, 'train'))
        
        
        self.train_list = self.get_patient_list(self.train_directory, patient_case)
        
        self.image_size = image_size + (1,)
        self.num_classes = 2
        
        self.augmentation = DataAugmentation(seed=1235)
        self.list_shuffle = random.Random(0)
        
    
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
        

        Parameters
        ----------
        patient_directory : Union[str, Path]
            DESCRIPTION.

        Returns
        -------
        patient_data : TYPE
            DESCRIPTION.

        """
        patient_data = {}
        image_list = []
        segmentation_list = []
        
        image_path = os.path.join(patient_directory, 'images')
        segmentation_path = os.path.join(patient_directory, 'segmentations')
        
        for file in sorted(os.listdir(image_path)):
            file_path = os.path.join(image_path, file)
            image = SegDataLoader.load_image(file_path)
            
            file_path = os.path.join(segmentation_path, file)
            segmentation = SegDataLoader.load_image(file_path)
            segmentation = segmentation == 1 # Only select Left Atrium labels
            
            image_list.append(image)
            segmentation_list.append(segmentation)
            
        
        patient_data['images'] = image_list
        patient_data['segmentations'] = segmentation_list
        
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
        

        Parameters
        ----------
        patient_directory : Union[str, Path]
            DESCRIPTION.

        Returns
        -------
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
            
            self.save_cache(patient_directory, patient_data)
            self.save_memory(patient_directory, patient_data)
        
        return patient_data
    

    @staticmethod
    def to_structure(image: sitk.Image, segmentation: sitk.Image,
                     add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Converts the input image and segmentation into a structured data
        format.

        Parameters
        ----------
        image : sitk.Image
            DESCRIPTION.
        segmentation : sitk.Image
            DESCRIPTION.
        add_batch_axis : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        data : Tuple[Dict[str, np.ndarray]]
            The structured data, represented as a tuple of dictionaries containing
            NumPy arrays.

        """
        dtype = np.float32
        image = SegDataLoader.to_numpy(image, dtype)
        segmentation = SegDataLoader.to_numpy(segmentation, dtype)
                
        if add_batch_axis:
            image = np.expand_dims(image, axis=0) 
            segmentation = np.expand_dims(segmentation, axis=0) 
        
        if image.ndim == 3:
           image = np.expand_dims(image, axis=-1) 
        if segmentation.ndim == 3:
           segmentation = np.expand_dims(segmentation, axis=-1) 
        
        data = ({'input_img': image},
                {'output_seg': segmentation})
            
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
        

    def augment_data(self, image: sitk.Image, segmentation: sitk.Image) -> Tuple[sitk.Image]:
        """
        

        Parameters
        ----------
        image : sitk.Image
            DESCRIPTION.
        segmentation : sitk.Image
            DESCRIPTION.

        Returns
        -------
        image : sitk.Image
            The augmented image.
        segmentation : sitk.Image
            The augmented segmentation.

        """
        image = self.augmentation.random_augmentation(image, is_labels=False,
                                                      likelihood=0.95, use_cache=False)
        segmentation = self.augmentation.random_augmentation(segmentation, is_labels=True,
                                                             likelihood=0.95, use_cache=True)
        
        return image, segmentation
        

    def data_generator(self, patient_directory: Union[Path, str], patient_indexes: List[int],
                       augment: bool = False, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a given patient directory.

        Parameters
        ----------
        patient_directory : Union[Path, str]
            The directory path of the patient data.
        patient_indexes : List[int]
            Which phases to be used as training.
        augment : bool, optional
            Whether to perform random augmentation. The default is False.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        if verbose > 0:
            print('Generating patient: ', patient_directory)
        patient_data = self.generator(patient_directory)
        
        for i in patient_indexes:
            image = patient_data['images'][i]
            segmentation = patient_data['segmentations'][i]
            
            if augment:
                image, segmentation = self.augment_data(image, segmentation)
                
            data = self.to_structure(image, segmentation, fp_32=self.fp_32)
            yield data
            self.clear_data(data)
            
        gc.collect()
    
    
    def data_generator_index(self, index: int, patient_indexes: List[int],
                             augment: bool = False, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a specific index in the dataset.

        Parameters
        ----------
        index : int
            The index of the patient data in the dataset.
        patient_indexes : List[int]
            Which phases to be used as training.
        augment : bool, optional
            Whether to perform random augmentation. The default is False.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        patient_diretory = self.train_list[index]
            
        yield from self.data_generator(patient_diretory, patient_indexes, augment=augment, verbose=verbose)
            
            
    def dataset_generator(self, patient_indexes: List[int], augment: bool = False,
                          shuffle: bool = False, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates data for a specific dataset type.

        Parameters
        ----------
        patient_indexes : List[int]
            Which phases to be used as training.
        augment : bool, optional
            Whether to perform random augmentation. The default is False.
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to False.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        data_list = self.train_list
            
        if shuffle:
            self.list_shuffle.shuffle(data_list)
            
        for patient_directory in data_list:
            yield from self.data_generator(patient_directory, patient_indexes, augment, verbose)
                
                
    def train_generator(self, augment: bool = True, shuffle: bool = True,
                        verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates training data.

        Parameters
        ----------
        augment : bool, optional
            Whether to perform random augmentation. The default is False.
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to False.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        patient_indexes = [0, 8, 15]
        yield from self.dataset_generator(patient_indexes=patient_indexes,
                                          augment=augment, shuffle=shuffle,
                                          verbose=verbose)
    
    
    def validation_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates validation data.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        patient_indexes = list(range(20))
        del patient_indexes[15]
        del patient_indexes[8]
        del patient_indexes[0]
        yield from self.dataset_generator(patient_indexes=patient_indexes,
                                          augment=False, shuffle=False,
                                          verbose=verbose)
            
    
    def test_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        """
        Generates test data.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Yields
        ------
        data : Tuple[Dict[str, np.ndarray]]
            A tuple containing a dictionary of structured data arrays.

        """
        patient_indexes = list(range(20))
        del patient_indexes[15]
        del patient_indexes[8]
        del patient_indexes[0]
        yield from self.dataset_generator(patient_indexes=patient_indexes,
                                          augment=False, shuffle=False,
                                          verbose=verbose)
    
    