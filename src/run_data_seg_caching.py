import shutil

from loader.data_generator_seg import SegDataLoader


def cache_data() -> None:
    """
    Saves the processed data.    

    Returns
    -------
    None
    
    """
    data_loader = SegDataLoader(memory_cache=False, disk_cache=True)
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(data_loader.cache_directory, ignore_errors=True)
    
    all_data = []
    all_data.extend(data_loader.train_list)
    
    # Cache data using class generator
    total_data = len(all_data)
    for i in range(len(all_data)):
        print('Processing ({}/{}): {}'.format(str(i+1), str(total_data), str(all_data[i])))
        data_loader.generator(all_data[i])
        

if __name__ == '__main__':
    cache_data()