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
    parser.add_argument('--mass', default=False, action='store_true')
    parser.add_argument('--multiple', default=False, action='store_true')
    parser.add_argument('--half', default=False, action='store_true')
    parser.add_argument('--left', default=2.0, type=float)
    parser.add_argument('--right', default=3.0, type=float)
    args = parser.parse_args()
        
    WORLD_SCALE = 10.0 # the scale of imagined world. Set positions and velocitys by this scale
    STRIDE = 10
    MIN_PRE_FRAMES = 8
    MIN_POST_FRAMES = 8
    NUM_FRAMES = 32
    HDF5_SIZE=1000
    FPS=5

    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.5
    MAX_R = 1.5


    # def out_square(r1, r2, v1, v2):
    #     assert MIN_R <= r1 <= MAX_R and MIN_V <= v1 <= MAX_V and \
    #         MIN_R <= r2 <= MAX_R and MIN_V <= v2 <= MAX_V
        
    #     in_square = (2.2 <= v1 <= 2.8 and 0.975 <= r1 <= 1.125 and 2.2 <= v2 <= 2.8 and 0.975 <= r2 <= 1.125) \
    #     or (1.2 <= v1 <= 1.6 and 1.2 <= r1 <= 1.3 and 1.2 <= v2 <= 1.6 and 1.2 <= r2 <= 1.3) \
    #     or (3.2 <= v1 <= 3.6 and 0.85 <= r1 <= 0.95 and 3.2 <= v2 <= 3.6 and 0.85 <= r2 <= 0.95)
        
    #     return not in_square
    
    def out_square_vel(r1, r2, v1, v2):
        assert MIN_R <= r1 <= MAX_R and MIN_V <= v1 <= MAX_V and \
            MIN_R <= r2 <= MAX_R and MIN_V <= v2 <= MAX_V
        
        # in_square = (2.0 <= v1 <= 3.0 and 2.0 <= v2 <= 3.0)     
        in_square = (args.left <= v1 <= args.right and args.left <= v2 <= args.right)     
        
        return not in_square
    
    def out_half_square_vel(r1, r2, v1, v2):
        assert MIN_R <= r1 <= MAX_R and MIN_V <= v1 <= MAX_V and \
            MIN_R <= r2 <= MAX_R and MIN_V <= v2 <= MAX_V
        
        
        in_train = (MIN_V <= v1 <= args.left and  MIN_V <= v2 <= MAX_V) or \
               (MIN_V <= v1 <= MAX_V and  MIN_V <= v2 <= args.left)
        
        return in_train
    
    
    def out_multiple_squares_vel(r1, r2, v1, v2):
        assert MIN_R <= r1 <= MAX_R and MIN_V <= v1 <= MAX_V and \
            MIN_R <= r2 <= MAX_R and MIN_V <= v2 <= MAX_V
            
        # [1.2, 2.2, 2.7, 3.7],
        # [2.7, 3.7, 1.2, 2.2],
        # [2.9, 3.5, 3.3, 3.9]
        
        in_square = (1.2 <= v1 <= 2.2 and 2.7 <= v2 <= 3.7) or \
                    (2.7 <= v1 <= 3.7 and 1.2 <= v2 <= 2.2) or \
                    (2.9 <= v1 <= 3.5 and 3.3 <= v2 <= 3.9)  
        
        return not in_square
    
    
    def out_square_mass(r1, r2, v1, v2):
        assert MIN_R <= r1 <= MAX_R and MIN_V <= v1 <= MAX_V and \
            MIN_R <= r2 <= MAX_R and MIN_V <= v2 <= MAX_V
        
        # in_square = (0.8 <= r1 <= 1.2 and 0.8 <= r2 <= 1.2)     
        in_square = (args.left <= r1 <= args.right and args.left <= r2 <= args.right)     
        
        return not in_square
    
    out_square = None
    if args.mass:
        out_square = out_square_mass
    elif args.half:
        out_square = out_half_square_vel
    else:
        if args.multiple:
            out_square = out_multiple_squares_vel
        else:
            out_square = out_square_vel
        
    # train data

    # Open the existing HDF5 file
    input_file_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/collision_1.6M_v1.hdf5'
    output_file_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out'
    if args.mass:
        output_file_path += f'_mass_{args.left}-{args.right}'
    elif args.half:
        output_file_path += f'half_vel_{args.left}'
    else:
        output_file_path += f'_vel_{args.left}-{args.right}'
        
    if args.multiple:
        output_file_path += '_multiple'

    # Initialize a counter for the total number of selected elements
    total_selected = 0
    total = 0

    # First pass: Count the number of elements that meet the criteria
    with h5py.File(input_file_path, "r") as tmp_f:
        # Loop through the elements in 'init_streams' to count the number of selected elements
        for elem_key in tqdm(tmp_f['init_streams'].keys()):
            init_data = tmp_f['init_streams'][elem_key]
            selected_indices = [i for i in range(init_data.shape[0]) if out_square(*init_data[i])]
            total_selected += len(selected_indices)
            total += init_data.shape[0]

    print(f"Total elements: {total}")
    print(f"Total elements selected: {total_selected}")
    output_file_path += f'_{get_size_str(total_selected)}.hdf5'
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
        for elem_key in tqdm(tmp_f['init_streams'].keys()):
            init_data = tmp_f['init_streams'][elem_key]
            selected_indices = [i for i in range(init_data.shape[0]) if out_square(*init_data[i])]
            
            # Collect the selected data in batches
            for i in selected_indices:
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


# python3 two_balls_collision_square_out.py --left 2.0 --right 3.0
# python3 two_balls_collision_square_out.py --left 1.75 --right 3.25
# python3 two_balls_collision_square_out.py --left 1.5 --right 3.5
# python3 two_balls_collision_square_out.py --left 1.25 --right 3.75
# python3 two_balls_collision_square_out.py --left 1.1 --right 3.9

# python3 two_balls_collision_square_out.py --mass --left 0.8 --right 1.2
# python3 two_balls_collision_square_out.py --mass --left 0.8 --right 1.2
# python3 two_balls_collision_square_out.py --multiple
# python3 two_balls_collision_square_out.py --left 1.25 --right 3.75 --half
