import math
import random
import numpy as np
from tqdm import tqdm_notebook
import phyre
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from functools import partial
import h5py
import imageio
import tempfile
# import imageio.v3 as iio # need python 3.9
from pathlib import Path
import time
from tqdm import tqdm
import imageio.v3 as iio # need python 3.9
from io import BytesIO
import torch
import os
import signal
import pandas as pd

def set_seeds(seed_value=0):
    """Set seeds for reproducibility."""
    random.seed(seed_value)  # Python's built-in random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # For CUDA


def normalize_probabilities(p1, p2, p3):
    """Normalize the probabilities to sum to 1."""
    total = p1 + p2 + p3
    return p1 / total, p2 / total, p3 / total



def read_file_to_byte_stream(file_path):
    """Read the entire content of a file into a byte stream."""
    with open(file_path, 'rb') as file:
        byte_stream = file.read()
    return byte_stream


def store_byte_stream_to_hdf5(byte_stream, hdf5_path, dataset_name):
    """Store a byte stream in an HDF5 file as a dataset."""
    data = np.frombuffer(byte_stream, dtype=np.uint8)
    
    with h5py.File(hdf5_path, 'w') as hdf:
        hdf.create_dataset(dataset_name, data=data, dtype='uint8')


def convert_frames_to_mp4_bytestream(frames, fps=5):
    """
    Convert a sequence of frames into an MP4 byte stream.

    Parameters:
    - frames: A list or array of frames (numpy arrays).
    - fps: Frames per second for the output video.

    Returns:
    - A byte stream of the MP4 video.
    """
    # Use a temporary file to store the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_name = temp_file.name      
        imageio.mimsave(temp_file_name, frames, fps=fps)

        # Read the MP4 file content into a byte stream
        with open(temp_file_name, 'rb') as video_file:
            mp4_bytestream = video_file.read()
    stream_array = np.frombuffer(mp4_bytestream, dtype='uint8')
    return stream_array



def convert_frames_to_mp4_bytestream_wo_disk(frames, fps=5):
    """
    Convert a sequence of frames into an MP4 byte stream, entirely in-memory.
    
    Parameters:
    - frames: A list or array of frames (numpy arrays).
    - fps: Frames per second for the output video.
    
    Returns:
    - A byte stream of the MP4 video.
    """
    # Create an in-memory bytes buffer
    with BytesIO() as buffer:
        # Use imageio to write frames to the buffer as an MP4 video
        iio.imwrite(buffer, frames, extension='.mp4')
        
        # Get the byte stream from the buffer
        mp4_bytestream = buffer.getvalue()
    
    return np.frombuffer(mp4_bytestream, dtype=np.uint8)



def store_mp4_to_hdf5(mp4_path, hdf5_path):
    stream = read_file_to_byte_stream(mp4_path)
    store_byte_stream_to_hdf5(stream, hdf5_path, dataset_name='task1')
    
    
def store_batched_frames_to_hdf5(all_frames, hdf5_path, task_name, states=None):
    with h5py.File(hdf5_path, 'w') as hdf:
        if states is not None:
            batched_actions = states['actions']
            batched_objects = states['objects']
            action_dset =hdf.create_dataset('action_streams', data=batched_actions)
            object_dset =hdf.create_dataset('object_streams', data=batched_objects)
        
        video_dset =hdf.create_dataset('video_streams', 
                           shape=(len(all_frames),),
                           dtype=h5py.vlen_dtype(np.dtype('uint8')),
                           ) 
        
        # batch_size = 100 # takes lots of memory
        batch_size = 25 
        for i in range(0, len(all_frames), batch_size):
            batched_frames = all_frames[i:i+batch_size]
            # speedup time bottleneck by parallelizing batch
            # start_time_step = time.time()
            batched_images = phyre.vis.batched_observations_to_frames(batched_frames)

            # batched_images = np.stack(batched_images, 0)
            # batched_images = WAD_COLORS[batched_images]
            # print(f"Time for obs2frames", time.time() - start_time_step, "seconds")
            
            for j, images in enumerate(batched_images):
                # start_time_step = time.time()
                # writing to disk is faster than in-memory operation
                # stream = convert_frames_to_mp4_bytestream(images) # time bottleneck
                stream = convert_frames_to_mp4_bytestream_wo_disk(images) # time bottleneck
                # print(f"Time for frames2bytestream", time.time() - start_time_step, "seconds")
                
                trial_id = i + j
                video_dset[trial_id] = stream

                # save visualization
                store_video = True
                FPS=10
                if store_video:
                    video_dir = Path(args.data_dir) / 'vis_videos'
                    if not os.path.exists(video_dir):
                        os.mkdir(video_dir)
                    actions = states['actions'][trial_id]
                    # mp4_path = video_dir / f'{task_name}_x{actions[0]:.4f}_r{actions[2]:.4f}_y{actions[1]:.4f}.mp4'
                    mp4_path = video_dir / f"x{states['objects'][0, 0, 2, 0]:.5f}_{task_name}.mp4"
                    frames = images
                    # print(frames.shape, frames.min(), frames.max())
                    imageio.mimsave(mp4_path, frames, fps=FPS)
        

def decode_hdf5_to_frames(hdf5_path, trial_index):
    """Decode video frames from a byte stream stored in an HDF5 file by first writing to a temporary file."""
        
    hdf = h5py.File(hdf5_path, 'r')
        
    byte_stream = hdf['video_streams'][trial_index]
    byte_obj = byte_stream.tobytes()
    # Use a temporary file to write the byte stream
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(byte_obj)
    
    # Now read the video from the temporary file
    with imageio.get_reader(temp_file_name, format='mp4') as reader:
        frames = [frame for frame in reader]
        fps = reader.get_meta_data()['fps']
    
    # imageio.mimsave('deocded_v1.mp4', frames, fps=fps)
    
    return np.array(frames), fps
    
def decode_hdf5_to_frames_wo_disk(hdf5_path, trial_index, format_hint='.mp4'):
    """Decode video frames from a byte stream stored in an HDF5 file directly in memory."""
    hdf = h5py.File(hdf5_path, 'r')

    byte_stream = hdf['video_streams'][trial_index]
    byte_obj = byte_stream.tobytes()
    # Decode frames directly from byte stream
    frames = iio.imread(byte_obj, index=None, extension=format_hint)
    # imageio.mimsave('deocded_v2.mp4', frames, fps=5)
    return frames

def sample_and_simulate(simulator, task_index, num_trials, task_name, candidate_solutions, max_loop_per_action=100):
    """
    combine generate_action and simulate to avoid repreated invalid action from generate_action
    p1, p2, p3: floats, probabilities for each action generation method.
    """
    batched_images = []
    batched_actions = []
    batched_objects = []
    initial_featurized_objects = simulator.initial_featurized_objects[task_index]
    featurized_objects = initial_featurized_objects.features
    is_valid_action = lambda x: simulator._action_mapper.action_to_user_input(x)[1]
    solve_cnt = 0
    
    # For 256x256 image, 1 pixel = 0.026 radius
    if task_name == '20000:000':
        actions = []
        for x in [0.45]:
            for r in np.linspace(0.20, 0.25, 101):
            # for r in np.linspace(0.20, 0.25, 3):
                action = np.array([x, 0.46, r])
                actions.append(action)

    elif task_name == '20001:000':
        actions = []
        for x in np.linspace(0.39, 0.41, 101):
        # for x in np.linspace(0.39, 0.41, 3):
            for r in [0.2]:
                action = np.array([x, 0.46, r])
                actions.append(action)
                
    # elif task_name == '20002:000':
    elif '20002' in task_name:
        actions = []
        for x in [0.5]:
            # for y in [0.5, 0.65, 0.8]:
            for y in [0.5]:
                # for r in np.linspace(0.35, 0.45, 101):
                for r in [0.4]:
                    action = np.array([x, y, r])
                    actions.append(action)

    elif '20003' in task_name:
        actions = []
        # for x in np.linspace(0.39, 0.41, 101):
        for x in [0.4]:
            for y in [0.5]:
                for r in [0.4]:
                    action = np.array([x, y, r])
                    actions.append(action)
    else:
        raise NotImplementedError(task_name)
    
    # print(initial_featurized_objects.features)
    
    for i, action in enumerate(actions):
        # Check if the action is valid
        if is_valid_action(action):
            # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
            # TIME = FRAMES / FPS = k / (STRIDE * FPS)
            # print(f'{action}')
            simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True
                        , stride=STRIDE,  max_simulation_steps=MAX_SIMULATION_STEPS)
            
            if not simulation.status.is_invalid(): 
                batched_images.append(simulation.images)
                batched_actions.append(action)
                batched_objects.append(simulation.featurized_objects.features)
                if simulation.status.is_solved():
                    solve_cnt += 1
            else:
                print('invalid status', task_name, i, action)
        else:
            print('invalid action', task_name, i, action)

        # if task_name == '20003:000' and action[1] == 0.5:
        # # and (0.4007 <= action[0] <= 0.4015) \
        #     print('*'*30)
        #     print(action)
        #     print(simulation.featurized_objects.features[0, -1, :2])
        # if task_name == '20002:000' and action[1] == 0.5:
        #     print('*'*30)
        #     print(action)
        #     print(simulation.featurized_objects.features[0, -1, :4])
        if  '20003' in task_name:
            print('*'*30)
            print(task_name)
            print(simulation.featurized_objects.features[0, :4, :4])

        
    # print(f'Success rate of sampled actions to solve the task: {solve_cnt/num_trials*100:.1f} ({num_trials} trials)')
    
    # log for diversity analysis
    batched_actions = np.stack(batched_actions, 0)
    batched_objects = np.stack(batched_objects, 0) # (tirals, frames, objects, 14)
    # print(batched_actions.shape, batched_objects.shape)


    states = {
        'initial_objects': featurized_objects,
        'actions': batched_actions,
        'objects': batched_objects,
    }
    
    return batched_images, states



def generate_videos_hdf5(tasks, action_tier, success_actions_dict, num_trials=100, data_dir='./', seed=42):
    # fix randomness
    set_seeds(args.seed)
    
    # Start measuring time for step 1
    # start_time_step1 = time.time()
    
    # Step 1: initialize simulator
    simulator = phyre.initialize_simulator(tasks, action_tier)
    # End measuring time for step 1 and print duration
    # print("Time for Step 1 (initialize simulator):", time.time() - start_time_step1, "seconds")
    

    for task_index, task_name in enumerate(tasks):
        
        # Step 2: simulate K trials
        batched_imgs, states = sample_and_simulate(simulator, task_index, num_trials, task_name, 
                                                   success_actions_dict[task_name] if success_actions_dict else None)

        # Step 3: save all trials' videos
        hdf5_path = Path(data_dir) / f'{task_name}.hdf5'
        store_batched_frames_to_hdf5(batched_imgs, hdf5_path, task_name=task_name, states=states)

        # For 20002/20003
        x_pos_list.append(states['objects'][0, 0, 2, 0])
        print(x_pos_list)


def merge_files(data_path, new_path=None, filter=None):
    if new_path is None:
        new_path = str(data_path).rstrip("/") + ".hdf5"
    fnames = [name for name in os.listdir(data_path) if name.endswith(".hdf5")]
    fnames = sorted(fnames, key=lambda x: int(''.join(x.split(".")[0].split(':'))))
    with h5py.File(os.path.join(data_path, fnames[0]), "r") as tmp_f:
        keys = list(tmp_f.keys())
    new_f = h5py.File(new_path, "w")
    for k in keys:
        new_f.create_group(k)
    for name in tqdm(fnames):
        if filter is not None and not filter(name):
            continue
        print("==> merging: ", name)
        with h5py.File(os.path.join(data_path, name), "r") as f:
            for k in f.keys():
                new_f[k].create_dataset("{}".format(name.split(".")[0]), data=f[k])
    print("==> saving to: ", new_path)
    new_f.close()


def main_single_process(args):
    # tasks = sorted([f'{20000+template_id}:{i:03d}' 
    #                     for template_id in range(4) for i in range(1)])
    # tasks = ['20003:000']
    tasks = sorted([f'{20003}:{i:03d}' 
                        for i in range(100)])

    action_tier= 'ball'
        
    print(f'Generate simulated video from {len(tasks)} tasks x {args.num_trials} trials with {args.num_workers} processes.')
    
    for task in tqdm(tasks):
        generate_videos_hdf5([task], action_tier, None, args.num_trials, args.data_dir, args.seed)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # 64 workers comsume a lot of memory, about 1100G
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--timeout_per_trial', type=float, default=2.0)
    parser.add_argument('--data_dir', type=str, default='/mnt/bn/bykang/phy-data/phyre_combination_data/ambiguity/v3.2')
    args = parser.parse_args()
    
    args.timeout = args.timeout_per_trial * args.num_trials # not used
    
    # fix randomness
    set_seeds(args.seed)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    fps = 5
    STRIDE = int(100 / fps) #keep  STRIDE=20
    MAX_SIMULATION_STEPS=1000 # max_frames=50, max duration=10 seconds



    x_pos_list = []
    if args.num_workers == 1:
        main_single_process(args) # for debug
    else:
        main(args)

    merge_files(args.data_dir)

    x_pos_list.sort()
    print(x_pos_list)


# python3 data_generator_ambiguity.py --num_workers 1
