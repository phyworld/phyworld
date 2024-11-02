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
    
    
def store_batched_frames_to_hdf5(all_frames, hdf5_path, states=None):
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

    return
        
    
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


# def generate_action(featurized_objects, num_balls, is_valid_action, task_name, random_action_sampler, num_sampled_actions=1, eps=0.01, max_loop_per_action=100, ):
#     """
#     Generate a valid action based on the specified probabilities and methods.
    
#     Parameters:
#     - featurized_objects: np.ndarray, the initial states of scene objects.
#     - p1, p2, p3: floats, probabilities for each action generation method.
#     - candidate_actions: list of actions (np.ndarray) to sample from.
#     - num_balls: int, number of user balls (1 or 2).
#     - is_valid_action: function, checks if the generated action is valid.
#     - eps: float, a small epsilon value to ensure a gap between the object and the scene's upper bound.
    
#     Returns:
#     - action: np.ndarray, the generated valid action.
#     """
#     action_dim = 3 * num_balls
#     filtered_object_id = []
#     min_radius, max_radius = 1.0, 0
#     for i in range(featurized_objects.shape[1]):
#         if featurized_objects[0, i, 4] == 1:
#             cur_radius = featurized_objects[0, i, 3]/2
#             min_radius = min(min_radius, cur_radius/2)
#             max_radius = max(max_radius, cur_radius*4.0)
#         if ignore_sticks and featurized_objects[0, i, 5] == 1: # bar
#             continue
#         filtered_object_id.append(i)
#     max_radius = min(max_radius, 1.0/RADIUS_SCALE)
#     if len(filtered_object_id) == 0:
#         filtered_object_id = list(range(featurized_objects.shape[1]))
    
#     sampled_actions = []
#     loop_cnt = 0
    
#     while True:
#         # Choose the method based on normalized probabilities
#         choice = np.random.choice(['random', 'positioning', 'candidate'], p=[p1, p2, p3])
        
#         if choice == 'random':
#             # Random sample from action space, adjust dimensions for one or two ball tier
#             action = np.random.rand(action_dim)
#             action[2] = np.random.uniform(min_radius, max_radius) * RADIUS_SCALE
#         elif choice == 'positioning': 
#             action = []
#             for _ in range(num_balls):
#                 # Calculate positioning above another object with constraints
#                 object_idx = random.sample(filtered_object_id, k=1)[0]
#                 # print(object_idx)
#                 object_info = featurized_objects[0, object_idx]  # Including diameter for radius calculation
#                 radius = object_info[3] / 2
#                 my_radius = np.random.uniform(min_radius, max_radius)
#                 max_distubance = radius + my_radius
#                 x_disturbance = np.random.uniform(-0.8 * max_distubance, 0.8 * max_distubance)
#                 y_lower_bound = object_info[1] + x_disturbance * 0.9
#                 y_upper_bound = 1.0 - radius - eps
#                 my_y_pos = np.random.uniform(y_lower_bound, y_upper_bound)
#                 action.extend([object_info[0] + x_disturbance, my_y_pos, my_radius * RADIUS_SCALE])
#             action = np.array(action)
#         else:  # 'candidate'
#             # Sample from the candidate action set, adjusting for one or two balls if necessary
#             action_idx = np.random.randint(0, len(candidate_solutions))
#             action = candidate_solutions[action_idx][:action_dim]
        
#         # Check if the action is valid
#         if is_valid_action(action):
#             sampled_actions.append(action)
        
#         loop_cnt += 1
#         if loop_cnt >= max_loop_per_action * num_sampled_actions:
#             print(f'Task {task_name} has sampled {loop_cnt} action, yet still has not collected {num_sampled_actions} valid actions.') 
#             action = random_action_sampler()
#             sampled_actions.append(action)
            
#         if len(sampled_actions) == num_sampled_actions:
#             return sampled_actions
            

# def simulate(simulator, task_index, num_trials, task_name):
#     batched_images = []
#     initial_featurized_objects = simulator.initial_featurized_objects[task_index]
#     valid_action_check = lambda x: simulator._action_mapper.action_to_user_input(x)[1]
#     random_action_sampler = lambda x : simulator.build_discrete_actlamion_space(max_actions=1)[0]
#     solve_cnt = 0
    
#     # print(initial_featurized_objects.features)
    
#     for i in range(num_trials):
#         loop_cnt = 0
#         while True:
            
#             action = generate_action(
#                 initial_featurized_objects.features,
#                 num_balls=int(simulator.action_space_dim/3),
#                 is_valid_action=valid_action_check,
#                 num_sampled_actions=1,
#                 task_name=task_name,
#                 random_action_sampler=random_action_sampler,
#             )[0]
            
#             # print(task_name, i, action) # to see if deterministic
            
            
#             # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
#             # TIME = FRAMES / FPS = k / (STRIDE * FPS)
#             # print(f'{action}')
#             simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True
#                                                    , stride=STRIDE,  max_simulation_steps=MAX_SIMULATION_STEPS)
            
#             if not simulation.status.is_invalid(): 
#                 # print(action)

#                 batched_images.append(simulation.images)
#                 if simulation.status.is_solved():
#                     # print('sovled. Frames is ', len(simulation.images))
#                     solve_cnt += 1
#                 break
            
#             loop_cnt += 1
#             if loop_cnt > 10:
#                 print(f'Task {task_name} has sampled {loop_cnt} action in outer loop') 
            
#     # print(f'Success rate of sampled actions to solve the task: {solve_cnt/num_trials*100:.1f} ({num_trials} trials)')
    
#     return batched_images


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
    num_balls = int(simulator.action_space_dim/3)
    solve_cnt = 0
    eps = 0.05
    
    if candidate_solutions is not None:
        new_p1, new_p2, new_p3 = normalize_probabilities(p1, p2, p3)
        new_p3 = min(len(candidate_solutions) / num_trials, new_p3)
        new_p1 = (1 - new_p3) / (new_p1 + new_p2) * new_p1
        new_p2 = (1 - new_p3) / (new_p1 + new_p2) * new_p2
        # print(new_p1, new_p2, new_p3)
    else:
        new_p1, new_p2, new_p3 = normalize_probabilities(p1, p2, 0)
    
    filtered_object_id = []
    min_radius, max_radius = 1.0, 0
    for i in range(featurized_objects.shape[1]):
        if featurized_objects[0, i, 4] == 1:
            cur_radius = featurized_objects[0, i, 3]/2
            min_radius = min(min_radius, cur_radius/2)
            max_radius = max(max_radius, cur_radius*4.0)
        if ignore_sticks and featurized_objects[0, i, 5] == 1 and np.abs(featurized_objects[0, i, 2]) < 0.05: # ignore horizon bar
            continue
        filtered_object_id.append(i)
        
    min_radius = max(min_radius, 0.1/RADIUS_SCALE)
    max_radius = min(max_radius, 1.0/RADIUS_SCALE)
    if len(filtered_object_id) == 0:
        filtered_object_id = list(range(featurized_objects.shape[1]))
    if min_radius == 1.0 or max_radius == 0:
        min_radius = 0.1/RADIUS_SCALE
        max_radius = 1.0/RADIUS_SCALE
        
    random_action_space = None
    
    # print(initial_featurized_objects.features)
    
    for i in range(num_trials):
        loop_cnt = 0
        while True:
            # if loop_cnt > 0 and loop_cnt % 100 == 0: print(loop_cnt)
            # if loop_cnt > 1000: return None, None
            # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
            # TIME = FRAMES / FPS = k / (STRIDE * FPS)
            # Choose the method based on normalized probabilities
            choice = np.random.choice(['random', 'positioning', 'candidate'], p=[new_p1, new_p2, new_p3], )
            if loop_cnt < max_loop_per_action:
                # heuristic rules
                if choice == 'random':
                    # Random sample from action space, adjust dimensions for one or two ball tier
                    action = np.random.rand(simulator.action_space_dim)
                    action[2] = np.random.uniform(min_radius, max_radius) * RADIUS_SCALE
                    if simulator.action_space_dim == 6:
                        action[5] = np.random.uniform(min_radius, max_radius) * RADIUS_SCALE
                elif choice == 'positioning': 
                    action = []
                    for _ in range(num_balls):
                        # Calculate positioning above another object with constraints
                        object_idx = random.sample(filtered_object_id, k=1)[0]
                        # print(object_idx)
                        object_info = featurized_objects[0, object_idx]  # Including diameter for radius calculation
                        radius = object_info[3] / 2
                        my_radius = np.random.uniform(min_radius, max_radius)
                        max_distubance = radius + my_radius
                        x_disturbance = np.random.uniform(-0.8 * max_distubance, 0.8 * max_distubance)
                        if np.random.rand() < 0.5:
                            # higher than target object
                            y_lower_bound = object_info[1] + x_disturbance * 0.9
                            y_upper_bound = 1.0 - radius - eps
                        else:
                            # lower than target object
                            y_lower_bound = radius + eps 
                            y_upper_bound = object_info[1] - x_disturbance * 0.9
                        
                        my_y_pos = np.random.uniform(y_lower_bound, y_upper_bound)
                        action.extend([object_info[0] + x_disturbance, my_y_pos, my_radius * RADIUS_SCALE])
                    action = np.array(action)
                else:  # 'candidate'
                    action_idx = np.random.randint(0, len(candidate_solutions))
                    action = candidate_solutions[action_idx]
                    # print(action[2], min_radius * RADIUS_SCALE)
                    action[2] = max(action[2], min_radius * RADIUS_SCALE)
                if simulator.action_space_dim == 6:
                    action[5] = max(action[5], min_radius * RADIUS_SCALE)
            else:
                # random sample valid actions
                print(f'Task {task_name} trial {i} has sampled {loop_cnt} action, yet still has not collected one valid action.') 
                if random_action_space is None:
                    random_action_space = simulator.build_discrete_action_space(max_actions=2*num_trials)       
                action = random.choice(random_action_space)
                action[2] = np.random.uniform(min_radius, max_radius) * RADIUS_SCALE
                if simulator.action_space_dim == 6:
                    action[5] = np.random.uniform(min_radius, max_radius) * RADIUS_SCALE
                choice = 'valid_random'
            
            # Check if the action is valid
            if is_valid_action(action):
                # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
                # TIME = FRAMES / FPS = k / (STRIDE * FPS)
                # print(f'{action}')
                simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True
                                                    , stride=STRIDE,  max_simulation_steps=MAX_SIMULATION_STEPS)
                
                if not simulation.status.is_invalid(): 
                    # print(task_name, choice, action)
                    
                    batched_images.append(simulation.images)
                    batched_actions.append(action)
                    batched_objects.append(simulation.featurized_objects.features)
                    if simulation.status.is_solved():
                        solve_cnt += 1
                    break
            
            # invalid action or invalid status
            loop_cnt += 1

            
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
        # Start measuring time for step 2
        # start_time_step2 = time.time()
        
        # Step 2: simulate K trials
        batched_imgs, states = sample_and_simulate(simulator, task_index, num_trials, task_name, 
                                                   success_actions_dict[task_name] if success_actions_dict else None)
        # End measuring time for step 2 and print duration
        # print(f"Time for Step 2 (simulate {num_trials} trials) for task", task_name, ":", time.time() - start_time_step2, "seconds")
        
        # Start measuring time for step 3
        # start_time_step3 = time.time()
        
        # Step 3: save all trials' videos
        hdf5_path = Path(data_dir) / f'{task_name}.hdf5'
        store_batched_frames_to_hdf5(batched_imgs, hdf5_path, states=states)
        # End measuring time for step 3 and print duration
        # print("Time for Step 3 (save all trials' videos) for task", task_name, ":", time.time() - start_time_step3, "seconds")

# Define the timeout handler
# def timeout_handler(signum, frame):
#     raise TimeoutError

# def generate_videos_hdf5_with_timeout(timeout, *args):
#     # Set the alarm signal and handler (only works on Unix/Linux)
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(max(1, int(timeout)))  # Set the timeout        
#     try:
#         generate_videos_hdf5(*args)
#         signal.alarm(0)  # Cancel the alarm
#     # except TimeoutError:
#     except Exception:
#         task, data_dir = args[0][0], args[3]
#         print(f"Task {task} timed out")
#         # remove unfinished file args.data_dir + task.hdf5
#         unfinished_file_path = os.path.join(data_dir, f"{task}.hdf5")
#         try:
#             if os.path.exists(unfinished_file_path):
#                 os.remove(unfinished_file_path)
#                 print(f"Removed unfinished file: {unfinished_file_path}")
#         except OSError as os_error:
#             print(f"Error removing file: {os_error}")   
#     finally:
#         signal.alarm(0)  # Ensure the alarm is canceled


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


def main(args):
    if args.run_id == -1:
        tasks = sorted([f'{10000+template_id}:{i:03d}' 
                        for template_id in range(70) for i in range(100)])
    elif 0 <= args.run_id <= 6:
        tasks = sorted([f'{10000+template_id}:{i:03d}' 
                        for template_id in range(args.run_id*10, (args.run_id+1)*10) for i in range(100)])
        if args.run_id == 6: # eval
            args.num_trials = 1
    else:
        raise ValueError        
    action_tier= 'ball'
        
    print(f'Generate simulated video from {len(tasks)} tasks x {args.num_trials} trials with {args.num_workers} processes.')
    
    futures_map = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for task in tasks:
            future = executor.submit(generate_videos_hdf5, [task], action_tier, None, args.num_trials, args.data_dir, args.seed)
            # Map the future to its input arguments
            futures_map[future] = task

        # Iterate over the futures as they complete
        for future in tqdm(as_completed(futures_map), total=len(futures_map), desc="Generating Videos"):
            result = future.result()  # Adjust timeout as needed
 

def main_single_process(args):
    if args.run_id == -1:
        tasks = sorted([f'{10000+template_id}:{i:03d}' 
                        for template_id in range(70) for i in range(100)])
    elif 0 <= args.run_id <= 6:
        tasks = sorted([f'{10000+template_id}:{i:03d}' 
                        for template_id in range(args.run_id*10, (args.run_id+1)*10) for i in range(100)])
        if args.run_id == 6: # eval
            args.num_trials = 1
    else:
        raise ValueError   
    action_tier= 'ball'
        
    print(f'Generate simulated video from {len(tasks)} tasks x {args.num_trials} trials with {args.num_workers} processes.')
    
    for task in tqdm(tasks):
        generate_videos_hdf5([task], action_tier, None, args.num_trials, args.data_dir, args.seed)


       
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_id', type=int, default=-1, help="""
        -1: generate all 70 templates.
        i (0 <= i <= 6): generate templates 10*i - 10*(i+1)
    """)
    # 64 workers comsume a lot of memory, about 1100G
    parser.add_argument('--num_workers', type=int, default=64)
    # total 70 templates
    parser.add_argument('--num_trials', type=int, default=1000)
    parser.add_argument('--timeout_per_trial', type=float, default=2.0)
    parser.add_argument('--data_dir', type=str, default='/mnt/bn/bykang/phy-data/phyre_combination_data/4_in_8')
    args = parser.parse_args()
    
    args.timeout = args.timeout_per_trial * args.num_trials # not used
    
    # fix randomness
    set_seeds(args.seed)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    p1 = 0.00 # random
    p2 = 1.0 # heuristic
    # p3 = 0.10 # stable solution 
    p3 = 0.0 # stable solution 
    ignore_sticks = True
    MAX_RADIUS = 0.2
    RADIUS_SCALE = 5.0
    candidate_solutions = []

    fps = 5
    STRIDE = int(100 / fps) #keep  STRIDE=20
    MAX_SIMULATION_STEPS=1000 # max_frames=50, max duration=10 seconds


    if args.num_workers == 1:
        main_single_process(args) # for debug
    else:
        main(args)

    # merge_files(args.data_dir)


# sudo /mnt/bn/yueyang/miniconda/envs/phyre2/bin/python3 data_generator_v2.py --num_workers 64
# python3 data_generator_v2.py --num_workers 64 --run_id -1
# python3 data_generator_v2.py --num_workers 64 --run_id $ARNOLD_ID --data_dir /mnt/bn/bykang/phy-data/phyre_combination_data/4_in_8/train
# python3 data_generator_v2.py --num_workers 64 --run_id 6 --data_dir /mnt/bn/bykang/phy-data/phyre_combination_data/4_in_8/eval

# python3 data_generator_v2.py --num_workers 64 --run_id -1 --num_trials 1 --data_dir /mnt/bn/bykang/phy-data/phyre_combination_data/4_in_8/vis