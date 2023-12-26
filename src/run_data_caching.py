from typing import Optional
import shutil

from loader.data_generator import BaseDataLoader


def cache_data(dilation_radius: int, data_type: Optional[str]) -> None:
    """
    Saves the processed data.

    Parameters
    ----------
    dilation_radius : int
        The dilation radius to apply to the contour.
    data_type : str, optional
        From which data folder to process the data. Default is None.

    Returns
    -------
    None

    """
    data_loader = BaseDataLoader(memory_cache=False, disk_cache=True,
                                 dilation_radius=dilation_radius,
                                 translation_alignment=True, data_type=data_type)
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(data_loader.cache_directory, ignore_errors=True)
    
    all_data = data_loader.train_list
    # Cache data using class generator
    total_data = len(all_data)
    for i in range(len(all_data)):
        print('Processing ({}/{}): {}'.format(str(i+1), str(total_data), str(all_data[i])))
        data_loader.generator(all_data[i])
        

if __name__ == '__main__':
    # For evaluation
    dilation_radius = 0
    data_type = None
    cache_data(dilation_radius, data_type)
    dilation_radius = 0
    data_type = 'nn'
    cache_data(dilation_radius, data_type)
    # For training
    dilation_radius = 2
    data_type = 'nn'
    cache_data(dilation_radius, data_type)