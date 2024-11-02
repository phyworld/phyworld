import numpy as np
import time
import imageio
from Box2D import *
import torch
import random
import h5py
import imageio.v3 as iio # need python 3.9
from io import BytesIO
import argparse
import os
import pathlib
from pathlib import Path
from tqdm import tqdm
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_size_str(size):
    """
    Convert a numerical size into a string format with either "K" (thousands) or "M" (millions),
    keeping only one decimal place.
    
    :param size: The numerical size to convert.
    :return: The size as a string formatted with "K" or "M".
    """
    if size < 1_000:
        return f"{size}"
    elif size < 1_000_000:
        return f"{size / 1_000:.1f}K"
    else:
        return f"{size / 1_000_000:.1f}M"


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False, action='store_true')
    args = parser.parse_args()
    
        
    WORLD_SCALE = 10.0 # the scale of imagined world. Set positions and velocitys by this scale
    STRIDE = 10
    NUM_MIN_FRAMES = 24
    NUM_FRAMES = 32
    HDF5_SIZE=1000
    FPS=5

    MIN_V = 1.0
    MIN_V2 = 1.5
    MID_V = 2.5
    MAX_V2 = 3.5
    MAX_V = 4.0
    
    MIN_R = 0.7
    MIN_R2 = 0.8
    MID_R = 1.05
    MAX_R2 = 1.3
    MAX_R = 1.4
    

    def in_square(r, v):
        assert MIN_R <= r <= MAX_R and MIN_V <= v <= MAX_V
        
        in_square = (v <= MID_V and r <= MID_R) \
        or (v >= MID_V and r >= MID_R)
        return in_square
    
    # deprecated
    # def eval_square(r, v):
    #     assert MIN_R <= r <= MAX_R and MIN_V <= v <= MAX_V
        
    #     in_square = (v <= MIN_V2 and r >= MAX_R2) \
    #     or (v >= MAX_V2 and r <= MIN_R2)
    #     return in_square
        

    # Open the existing HDF5 file
    input_file_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/data_300.0K_in_dist_v2.hdf5'
    output_file_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/size_vel'

    # Initialize a counter for the total number of selected elements
    total_selected = []

    # First pass: Count the number of elements that meet the criteria
    with h5py.File(input_file_path, "r") as tmp_f:
        # Loop through the elements in 'init_streams' to count the number of selected elements
        for elem_key in tqdm(tmp_f['init_streams'].keys()):
            init_data = tmp_f['init_streams'][elem_key]
            if not args.eval:
                selected_indices = [(elem_key, i) for i in range(init_data.shape[0]) if in_square(init_data[i, 0], init_data[i, 1])]
            else:
                selected_indices = [(elem_key, i) for i in range(init_data.shape[0]) if eval_square(init_data[i, 0], init_data[i, 1])]
            total_selected.extend(selected_indices)

    if not args.eval:
        print(f"Total elements selected: {get_size_str(len(total_selected))}")
        output_file_path += f'_{get_size_str(len(total_selected))}.hdf5'
    else:
        total_selected = random.sample(total_selected, k=800)
        print(f"Total elements selected: {get_size_str(len(total_selected))}")
        output_file_path += f'_eval_{get_size_str(len(total_selected))}.hdf5'
    print(output_file_path)

    # Second pass: Write the selected elements to the new HDF5 file
    with h5py.File(input_file_path, "r") as tmp_f, h5py.File(output_file_path, "w") as new_f:
        # Create new groups for each key in the new file
        new_f.create_group('init_streams')
        new_f.create_group('position_streams')
        new_f.create_group('video_streams')
        
        # Initialize batch counters
        batch_counter = 0
        element_counter = 0
        
        # Temporary lists to collect elements for each batch
        init_batch = []
        position_batch = []
        video_batch = []

        # Iterate through the keys (which represent groups in HDF5)
        # for elem_key in tqdm(tmp_f['init_streams'].keys()):
        for elem_key, i in tqdm(total_selected):
            init_data = tmp_f['init_streams'][elem_key]
            
            # Get the corresponding data from each stream
            init_batch.append(tmp_f['init_streams'][elem_key][i])
            position_batch.append(tmp_f['position_streams'][elem_key][i])
            video_batch.append(tmp_f['video_streams'][elem_key][i])
            
            # Check if we've reached the batch size limit
            if len(init_batch) == 1000:
                # Write the batch to the new file
                dataset_name = f"{batch_counter:05d}"
                new_f['init_streams'].create_dataset(dataset_name, data=np.array(init_batch))
                new_f['position_streams'].create_dataset(dataset_name, data=np.array(position_batch))
                video_set = new_f['video_streams'].create_dataset(dataset_name, 
                                                        shape=len(video_batch),
                                                        dtype=h5py.vlen_dtype(np.dtype('uint8')),
                                                        )
                for j in range(0, len(video_batch)):
                    video_set[j] = video_batch[j]

                # Reset batches and increment batch counter
                init_batch = []
                position_batch = []
                video_batch = []
                batch_counter += 1

        # Write any remaining data that didn't fill a complete batch
        if len(init_batch) > 0:
            dataset_name = f"{batch_counter:05d}"
            new_f['init_streams'].create_dataset(dataset_name, data=np.array(init_batch))
            new_f['position_streams'].create_dataset(dataset_name, data=np.array(position_batch))
            video_set = new_f['video_streams'].create_dataset(dataset_name, 
                                                    shape=len(video_batch),
                                                    dtype=h5py.vlen_dtype(np.dtype('uint8')),
                                                    )
            for j in range(0, len(video_batch)):
                video_set[j] = video_batch[j]

# python3 one_ball_uniform_motion_size_vel.py