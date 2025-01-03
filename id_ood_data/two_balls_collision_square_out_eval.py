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

def set_seeds(seed_value=0):
    """Set seeds for reproducibility."""
    random.seed(seed_value)  # Python's built-in random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # For CUDA


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

def merge_files(data_path, new_path=None):
    if new_path is None:
        new_path = str(data_path).rstrip("/") + ".hdf5"
    fnames = [name for name in os.listdir(data_path) if name.endswith(".hdf5")]
    fnames = sorted(fnames, key=lambda x: int(x.split(".")[0]))
    with h5py.File(os.path.join(data_path, fnames[0]), "r") as tmp_f:
        keys = list(tmp_f.keys())
    new_f = h5py.File(new_path, "w")
    for k in keys:
        new_f.create_group(k)
    for name in tqdm(fnames):
        with h5py.File(os.path.join(data_path, name), "r") as f:
            for k in f.keys():
                new_f[k].create_dataset("{:05d}".format(int(name.split(".")[0])), data=f[k])
    print("==> saving to: ", new_path)
    new_f.close()

def store_mp4_to_hdf5(mp4_path, hdf5_path):
    stream = read_file_to_byte_stream(mp4_path)
    store_byte_stream_to_hdf5(stream, hdf5_path, dataset_name='task1')
    
    
def store_batched_frames_to_hdf5(all_frames, batched_postions, batched_radius, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf:
        _ =hdf.create_dataset('position_streams', data=batched_postions)
        _ =hdf.create_dataset('init_streams', data=batched_radius)
        video_dset =hdf.create_dataset('video_streams', 
                           shape=(len(all_frames),),
                           dtype=h5py.vlen_dtype(np.dtype('uint8')),
                           ) 
        
        for i in range(0, len(all_frames)):
            frames = all_frames[i]
            stream = convert_frames_to_mp4_bytestream_wo_disk(frames) # time bottleneck
            video_dset[i] = stream

    return


def uniform_random(low, high):
    return np.random.rand() * (high - low) + low


import numpy as np

# @jit(nopython=True)
def vectorized_draw_balls(positions, radii, colors=[(255, 0, 0), (0, 0, 255)], width=512, height=512):
    """
    Draw two balls with different colors on generated frames.
    
    :param n_frames: Number of frames to generate.
    :param width: Width of each frame.
    :param height: Height of each frame.
    :param positions: A numpy array of shape (n_frames, 2, 2), the x, y positions of two balls in each frame.
    :param radii: A numpy array of shape (2,), the fixed radii of the two balls.
    :param colors: A list of tuples defining the RGB colors of the two balls.
    """
    # Initialize frames
    

    n_frames = positions.shape[0]
    frames = np.ones((n_frames, height, width, 3), dtype=np.uint8) * 255
    colors = np.array(colors)

    # Prepare meshgrid for coordinates, normalize to [0, 1] range
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    # Adjust positions and radii for aspect ratio and normalized coordinate system
    norm_positions = positions
    norm_radii = radii

    # Calculate squared distances for each ball in normalized coordinates
    for i in range(2):  # Iterate over the two balls
        # stime = time.time()
        dx = X - norm_positions[:, i, 0, np.newaxis, np.newaxis]
        dy = Y - norm_positions[:, i, 1, np.newaxis, np.newaxis]

        
        distance_squared = dx**2 + dy**2
        radius_squared = norm_radii[i]**2
        # Calculate a soft mask based on distance to the circle's edge
        edge_width = 0.03  # Control the width of the edge transition
        soft_mask = np.clip((radius_squared - distance_squared) / (radius_squared * edge_width), 0, 1)[:,:,:,None]
        
        # print(soft_mask.shape, frames.shape, colors[i].shape)
        frames = (1 - soft_mask) * frames + soft_mask * colors[i]

        # print('preallocate', time.time() - stime)
        
    return frames.astype(np.uint8)


def simulate_two_balls_collision_in_one_dim(
    v1, v2, r1, r2, 
    world_scale,
    x1=None, x2=None, ball_height=None, 
    img_width=256, img_height=256,
    timestep=0.01, numsteps=500, stride=10,
    eps = 0.05,
    ):
    assert r1 > 0 and r2 > 0 and r1 + r2 < 0.5 * world_scale, f"{r1=}, {r2=}, {world_scale=}"

    if x1 is None or x2 is None: 
        # ensure we have at least MIN_PRE_FRAMES frames before collision
        min_pre_dist = MIN_PRE_FRAMES * (v1 + v2) * timestep * stride + r1 + r2
        # ensure we have at least MIN_POST_FRAMES frames after collision
        max_pre_dist = (numsteps / stride - MIN_POST_FRAMES) * timestep * stride * (v1 + v2) + r1 + r2
        # print(f'{min_pre_dist=}, {max_pre_dist=}')

        num_try = 0
        while num_try < 50:
            # when r1 is much larger than r2, constrain x1 at left to ensure ball2 have enough frames after collision
            if r1 > 2 * r2:
                max_pre_dist = min(min_pre_dist * 1.25, world_scale - r1 - r2)
                dist = uniform_random(min_pre_dist, max_pre_dist)
                x1_left = r1
                x1_right = x1_left + eps * world_scale

            if r2 > 2 * r1:
                max_pre_dist = min(min_pre_dist * 1.25, world_scale - r1 - r2)
                dist = uniform_random(min_pre_dist, max_pre_dist)
                x1_right = world_scale - r2 - dist
                x1_left = x1_right - eps * world_scale

            else:
                max_pre_dist = min(max_pre_dist, world_scale - r1 - r2)
                dist = uniform_random(min_pre_dist, max_pre_dist)
                x1_left = r1
                x1_right = world_scale - r2 - dist

            # ensure the whole body of ball in vision region for convieience when eval
            x1 = max(uniform_random(x1_left, x1_right), r1)
            x2 = min(x1 + dist, world_scale - r2)
            
            if v1 + v2 > 0:
                actual_pre_frames = (x2 - x1 - r1 - r2) / (v1 + v2) / (timestep * stride)
            else:
                actual_pre_frames = MIN_PRE_FRAMES + 1
                
            if actual_pre_frames < MIN_PRE_FRAMES:
                num_try += 1
                continue
            else:
                break
        
        success = True
        if num_try >= 50:
            print(f'{r1=}, {r2=}, {v1=}, {v2=} failed to generate required scene')
            success = False


        # print(f'{x1=}, {x2=}, {r1=}, {r2=}')

    if ball_height is None: 
        max_r = max(r1, r2)
        ball_height = uniform_random(
            max_r + eps * world_scale,
            (1 - eps) * world_scale - max_r,
        )

    world = b2World(gravity=(0, 0), doSleep=False)
    ball1 = world.CreateDynamicBody(position=(x1, ball_height), linearVelocity=(v1, 0))
    ball1.CreateCircleFixture(radius=r1, density=1, restitution=1.0)

    ball2 = world.CreateDynamicBody(position=(x2, ball_height), linearVelocity=(-v2, 0))
    ball2.CreateCircleFixture(radius=r2, density=1, restitution=1.0)

    ball1_xy, ball2_xy = [], []
    for i in range(numsteps):
        world.Step(timestep, 15, 20)
        if i % stride == 0:
            ball1_xy.append(np.array(ball1.position))
            ball2_xy.append(np.array(ball2.position))
        # print(ball1.position, ball2.position)


    ball1_xy = np.array(ball1_xy)
    ball2_xy = np.array(ball2_xy)

    positions = np.stack((ball1_xy, ball2_xy), axis=1)
    radii = np.array([r1, r2])
    # print(positions)

    # normlazied
    normed_positions = positions / world_scale
    normed_radii = radii / world_scale

    # print(positions[:,1,0]-positions[:,0,0])
    # print(r1+r2)


    frames = vectorized_draw_balls(normed_positions, normed_radii, width=img_width, height=img_height)
    frames = frames[:, ::-1, :, :]

    return frames, positions, success

def generate_videos_hdf5(scenes, scenes_id, data_dir='./', seed=42, store_video=False):
    # fix randomness
    set_seeds(args.seed)

    batched_images = []
    batched_positions = []
    batched_init = []

    for scene in scenes:
        r1, r2, v1, v2 = scene
        frames, positions, success = simulate_two_balls_collision_in_one_dim(v1, v2, r1, r2, world_scale=WORLD_SCALE, stride=STRIDE, numsteps=STRIDE*NUM_FRAMES)
        if not success: continue
        batched_images.append(frames)
        batched_positions.append(positions)
        batched_init.append([r1, r2, v1, v2])

        if store_video:
            video_dir = Path(data_dir) / 'vis_videos'
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            mp4_path = video_dir / f'{r1:.3f}_{r2:.3f}_{v1:.3f}_{v2:.3f}.mp4'
            # print(mp4_path)
            imageio.mimsave(mp4_path, frames, fps=FPS)
    
    batched_positions = np.stack(batched_positions, 0) # (1000, FRAMES, BALLS, XY)
    batched_init = np.array(batched_init)
    # print(f"batched_positions.shape: {batched_positions.shape}, batched_init.shape: {batched_init.shape}")

    hdf5_path = Path(data_dir) / f'{scenes_id}.hdf5'
    store_batched_frames_to_hdf5(batched_images, batched_positions, batched_init, hdf5_path)


    return


def main_single_process(combinations):
    process_tasks = [combinations[i:i + HDF5_SIZE] for i in range(0, len(combinations), HDF5_SIZE)]
    for task_id, task in tqdm(enumerate(process_tasks)):
        generate_videos_hdf5(task, task_id, data_dir=args.data_dir, seed=args.seed)

def main_vis(combinations):
    process_tasks = [combinations[i:i + HDF5_SIZE] for i in range(0, len(combinations), HDF5_SIZE)]
    for task_id, task in tqdm(enumerate(process_tasks)):
        generate_videos_hdf5(task, task_id, data_dir=args.data_dir, seed=args.seed, store_video=True)


def main(combinations):

    process_tasks = [combinations[i:i + HDF5_SIZE] for i in range(0, len(combinations), HDF5_SIZE)]

    futures_map = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for task_id, task in enumerate(process_tasks):
            # generate_videos_hdf5(task, task_id, data_dir=args.data_dir, seed=args.seed)
            future = executor.submit(generate_videos_hdf5, task, task_id, args.data_dir, args.seed)
            futures_map[future] = task_id

        # Iterate over the futures as they complete
        for future in tqdm(as_completed(futures_map), total=len(futures_map), desc="Generating Videos"):
            result = future.result()  # Adjust timeout as needed

def generate_combinations(r1_list, r2_list, v1_list, v2_list):
    # Generate all possible combinations of r1, r2, v1, and v2
    combinations = []
    for r1 in r1_list:
        for r2 in r2_list:
            for v1 in v1_list:
                for v2 in v2_list:
                    # Append the combination as a tuple to the list
                    combinations.append((r1, r2, v1, v2))
                
    np.random.shuffle(combinations) # in-place return None
    return combinations

def in_dist_generate():

    V_SPACE= 61
    R_SAPCE= 21
    
    if args.mass:
        MIN_V = 1.0
        MAX_V = 4.0
        MIN_R = 0.5
        MAX_R = 1.5
        MIN_OOD_R = 0.5 # avoid too small ball to disappear
        MAX_OOD_R = 2.0

        v_list = np.linspace(MIN_V, MAX_V, 13)
        r1_list = np.linspace(MIN_OOD_R, MAX_OOD_R, 7)
        r2_list = np.linspace(MIN_R, MAX_R, 5)
        
         # List to hold all combinations
        combinations = generate_combinations(r1_list, r2_list, v_list, v_list)

        
    else:
        OOD_MIN_V = 0.0
        OOD_MAX_V = 4.5
        MIN_V = 1.0
        MAX_V = 4.0
        MIN_R = 0.5
        MAX_R = 1.5

        v1_list = np.linspace(OOD_MIN_V, OOD_MAX_V, 21)
        v2_list = np.linspace(MIN_V, MAX_V, 13)
        r_list = np.linspace(MIN_R, MAX_R, 5)

        # List to hold all combinations
        combinations = generate_combinations(r_list, r_list, v1_list, v2_list)

    return combinations


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mass', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    
    # fix randomness
    set_seeds(args.seed)
        
    WORLD_SCALE = 10.0 # the scale of imagined world. Set positions and velocitys by this scale
    STRIDE = 10
    MIN_PRE_FRAMES = 4
    MIN_POST_FRAMES = 8
    NUM_FRAMES = 32
    HDF5_SIZE=100
    FPS=5

    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.5
    MAX_R = 1.5

    combinations = in_dist_generate()
    size = get_size_str(len(combinations))
    if args.mass:
        args.data_dir = pathlib.Path(f'/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_mass_{size}')
    else:
        args.data_dir = pathlib.Path(f'/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_vel_{size}')
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if args.num_workers == 1:
        main_single_process(combinations) # for debug
    else:
        main(combinations)

    merge_files(args.data_dir)


# python3 two_balls_collision_square_out_eval.py  --num_workers 69
# python3 two_balls_collision_square_out_eval.py --mass  --num_workers 69
