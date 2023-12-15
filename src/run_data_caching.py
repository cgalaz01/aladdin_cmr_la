import shutil

from loader.data_generator import BaseDataLoader


def cache_data() -> None:
    contour_properties = {
        'dilation_radius': 1,
        'dilate_segmentation': True}
    data_type = None #'nn'
    contour_properties = None
    data_loader = BaseDataLoader(memory_cache=False, disk_cache=True,
                                 contour_properties=contour_properties,
                                 translation_alignment=False, data_type=data_type)
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(data_loader.cache_directory, ignore_errors=True)
    
    all_data = []
    all_data.extend(data_loader.train_list)
    all_data.extend(data_loader.validation_list)
    all_data.extend(data_loader.test_list)
    
    # Cache data using class generator
    total_data = len(all_data)
    for i in range(len(all_data)):
        print('Processing ({}/{}): {}'.format(str(i+1), str(total_data), str(all_data[i])))
        data_loader.generator(all_data[i])
        

if __name__ == '__main__':
    cache_data()