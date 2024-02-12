import os
import pickle

import numpy as np

from scipy.ndimage import zoom

def resize_data(data_dir, input_size, file_paths, zoom_ratio=0.25):
    """
    Resize and save data volumes.
    
    Args:
        data_dir (str): The directory to save resized data.
        input_size (tuple): The original size of the input volumes.
        file_paths (list): List of paths to input data volumes.
        zoom_ratio (float, optional): Zooming ratio for resizing (default is 0.25).
    
    Returns:
        None
    """
    
    zoomed_dim = tuple([int(dim * zoom_ratio) for dim in input_size] + [1])

    for i, file_path in enumerate(file_paths):
        
        new_name = os.path.join(data_dir, f'./zoom_{zoom_ratio}_data/'+file_path.split('/')[-1])
        
        if not os.path.exists(new_name):
        
            data = np.ones(zoomed_dim)

            with open(file_path, 'rb') as handle:
                volume = pickle.load(handle)

            volume = zoom(volume, zoom=zoom_ratio, order=1)
            padding_start = int((zoomed_dim[0] - volume.shape[0]) / 2)

            data[padding_start: padding_start + volume.shape[0]] = volume[..., None]

            data = 1 - data

            print(f"{i/len(file_paths) * 100:0.2f} %", end='\r')

            with open(new_name, 'wb') as handle:
                pickle.dump(data, handle)