import os
import h5py
import numpy as np
from tqdm import tqdm

def split_hdf5_file(input_file, output_dir, num_parts=6):
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(input_file, 'r') as f:
        # Get the keys of the groups
        action_streams = f['action_streams']
        object_streams = f['object_streams']
        video_streams = f['video_streams']
        
        total_datasets = len(action_streams)
        print(f"Total datasets: {total_datasets}")
        
        # Determine the number of datasets per split
        assert total_datasets % num_parts == 0, "The total number of datasets is not evenly divisible by num_parts."
        
        datasets_per_part = total_datasets // num_parts
        meta_keys = list(action_streams.keys())
        
        for idx in range(num_parts):
            if idx <= 3: continue
            start = idx * datasets_per_part
            end = (idx + 1) * datasets_per_part
            out_file = os.path.join(output_dir, f'template_{idx*10:02d}:{(idx+1)*10-1:02d}.hdf5')
            
            with h5py.File(out_file, 'w') as out_f:
                # Create groups for each dataset category
                out_f.create_group('action_streams')
                out_f.create_group('object_streams')
                out_f.create_group('video_streams')
                
                # Write 1000 action_streams, object_streams, and video_streams to each file
                for i in tqdm(range(start, end), desc=f'Writing to {out_file}'):
                    
                    index = meta_keys[i]
                    action_data = action_streams[index]
                    object_data = object_streams[index]
                    video_data = video_streams[index]
                    
                    # Create datasets in the new file
                    out_f['action_streams'].create_dataset(index, data=action_data)
                    out_f['object_streams'].create_dataset(index, data=object_data)
                    out_f['video_streams'].create_dataset(index, data=video_data)
            
            print(f"Created {out_file} with datasets from {meta_keys[start]} to {meta_keys[end-1]}")

# Usage
input_file = '/mnt/bn/bykang/phy-data/phyre_combination_data/4_in_8/train_6M.hdf5'
output_dir = '/mnt/bn/yueyang/phy-data/phyre_combination_data/4_in_8/'  # Replace with your desired output directory
split_hdf5_file(input_file, output_dir)
