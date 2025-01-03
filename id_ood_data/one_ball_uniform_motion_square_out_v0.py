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
    dx = X - norm_positions[:, 0, np.newaxis, np.newaxis]
    dy = Y - norm_positions[:, 1, np.newaxis, np.newaxis]
    
    distance_squared = dx**2 + dy**2
    radius_squared = norm_radii**2
    # Calculate a soft mask based on distance to the circle's edge
    edge_width = 0.03  # Control the width of the edge transition
    soft_mask = np.clip((radius_squared - distance_squared) / (radius_squared * edge_width), 0, 1)[:,:,:,None]
    
    # print(soft_mask.shape, frames.shape, colors[i].shape)
    frames = (1 - soft_mask) * frames + soft_mask * colors[0]

    return frames.astype(np.uint8)


def simulate_one_ball_uniform_motion_in_one_dim(
    v1, r1,
    world_scale,
    x1=None, ball_height=None, 
    img_width=256, img_height=256,
    timestep=0.01, numsteps=500, stride=10,
    eps = 0.05,
    ):
    assert r1 > 0 and r1 < 0.25 * world_scale, f"{r1=}, {world_scale=}"

    if x1 is None: 
        # ensure we have at least MIN_PRE_FRAMES frames before out of vision
        min_dist = NUM_MIN_FRAMES * v1 * timestep * stride + r1
        max_init = world_scale - min_dist
        if max_init < r1:
            # print(f"{max_init=}, {r1=}, {v1=}")
            max_init = r1 + uniform_random(0, 0.5)
        x1 = uniform_random(r1, max_init)

    if ball_height is None: 
        max_r = r1
        ball_height = uniform_random(
            max_r + eps * world_scale,
            (1 - eps) * world_scale - max_r,
        )

    world = b2World(gravity=(0, 0), doSleep=False)

    if DIRECTION == 'right':
        ball1 = world.CreateDynamicBody(position=(x1, ball_height), linearVelocity=(v1, 0))
        ball1.CreateCircleFixture(radius=r1, density=1, restitution=1.0)
    elif DIRECTION == 'left':
        ball1 = world.CreateDynamicBody(position=(world_scale - x1, ball_height), linearVelocity=(-v1, 0))
        ball1.CreateCircleFixture(radius=r1, density=1, restitution=1.0)
    elif DIRECTION == 'upper':
        ball1 = world.CreateDynamicBody(position=(ball_height, x1), linearVelocity=(0, v1))
        ball1.CreateCircleFixture(radius=r1, density=1, restitution=1.0)
    elif DIRECTION == 'down':
        ball1 = world.CreateDynamicBody(position=(ball_height, world_scale - x1), linearVelocity=(0, -v1))
        ball1.CreateCircleFixture(radius=r1, density=1, restitution=1.0)

    ball1_xy = []
    for i in range(numsteps):
        world.Step(timestep, 15, 20)
        if i % stride == 0:
            ball1_xy.append(np.array(ball1.position))

    positions = np.array(ball1_xy)
    radii = np.array([r1])
    # print(positions)

    # normlazied
    normed_positions = positions / world_scale
    normed_radii = radii / world_scale

    frames = vectorized_draw_balls(normed_positions, normed_radii, width=img_width, height=img_height)
    frames = frames[:, ::-1, :, :]

    return frames, positions, True

def generate_videos_hdf5(scenes, scenes_id, data_dir='./', seed=42, store_video=False):
    # fix randomness
    set_seeds(args.seed)

    batched_images = []
    batched_positions = []
    batched_init = []

    for scene in scenes:
        r1, v1 = scene
        frames, positions, success = simulate_one_ball_uniform_motion_in_one_dim(v1, r1, world_scale=WORLD_SCALE, stride=STRIDE, numsteps=STRIDE*NUM_FRAMES)
        if not success: continue
        batched_images.append(frames)
        batched_positions.append(positions)
        batched_init.append([r1, v1])

        if store_video:
            video_dir = Path(data_dir) / 'vis_videos'
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            mp4_path = video_dir / f'{r1:.3f}_{v1:.3f}.mp4'
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

def generate_combinations(r1_list, v1_list):
    # Generate all possible combinations of r1, r2, v1, and v2
    combinations = []
    for r1 in r1_list:
        for v1 in v1_list:
            # Append the combination as a tuple to the list
            combinations.append((r1, v1))
            
    np.random.shuffle(combinations) # in-place return None
    return combinations

def in_dist_generate():
    # training or in-distribution eval
    
    if not args.extrapolate:
        if not args.eval:
            V_SPACE= 1000
            R_SPACE = 300
            # for larger square out to compensate the number of traning data
            # V_SPACE= 1500
            # R_SPACE = 450
        else:
            V_SPACE= 121
            R_SPACE = 36
        v_list = np.linspace(MIN_V, MAX_V, V_SPACE)
        r_list = np.linspace(MIN_R, MAX_R, R_SPACE)
        combinations = generate_combinations(r_list, v_list)
        
    else:
        
        R_SPACE = 36
        r_list = np.linspace(MIN_R, MAX_R, R_SPACE)
        
        MIN_OOD_V = 0
        MAX_OOD_V = 1.0
        # left_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 81)
        left_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 41)
        
        MIN_OOD_V = 4.0
        MAX_OOD_V = 6.0
        right_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 81)
        
        combinations = generate_combinations(r_list, left_v_ood_list) + \
            generate_combinations(r_list, right_v_ood_list)

    # List to hold all combinations

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
    parser.add_argument('--num_workers', type=int, default=64)
    # parser.add_argument('--data_ood_level', type=int) # data ood level for ood eval
    parser.add_argument('--data_for_vis', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--extrapolate', default=False, action='store_true')
    parser.add_argument('--left', default=2.0, type=float)
    parser.add_argument('--right', default=3.0, type=float)
    parser.add_argument('--n_train', default=-1, type=int)
    args = parser.parse_args()
    
    
    # fix randomness
    set_seeds(args.seed)
        
    WORLD_SCALE = 10.0 # the scale of imagined world. Set positions and velocitys by this scale
    STRIDE = 10
    NUM_MIN_FRAMES = 24
    NUM_FRAMES = 32
    FPS=5

    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4

    def in_square_v0(r, v):
        # return (2.2 <= v <= 2.8 and 0.975 <= r <= 1.125) \
        # or (1.2 <= v <= 1.6 and 1.2 <= r <= 1.3) \
        # or (3.2 <= v <= 3.6 and 0.85 <= r <= 0.95)
        return (2.2 <= v <= 2.8) \
        or (1.2 <= v <= 1.6) \
        or (3.2 <= v <= 3.6)
        
    def in_square_n(r, v):
        if args.n_train == 3:
            out_square = in_train = (1.0 <= v <= 1.25) or (2.375 <= v <= 2.625) or (3.75 <= v <= 4.0)
        if args.n_train == 4:
            out_square = in_train = (1.0 <= v <= 1.25) or (1.875 <= v <= 2.125) or (2.875 <= v <= 3.125) or (3.75 <= v <= 4.0)
        return not in_train

    DIRECTION = 'right'
    
    def in_square_left_right(r, v):
        # return (2.2 <= v <= 2.8 and 0.975 <= r <= 1.125) \
        # or (1.2 <= v <= 1.6 and 1.2 <= r <= 1.3) \
        # or (3.2 <= v <= 3.6 and 0.85 <= r <= 0.95)
        return args.left <= v <= args.right
    
    in_square = None
    
    if args.n_train >= 3:
        in_square = in_square_n
    else:
        in_square = in_square_left_right

    DIRECTION = 'right'
    

    data_dir = pathlib.Path(f'/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion')

    # training data
    combinations = in_dist_generate()
    # split in_square and out_square
    in_square_combinations = []
    out_square_combinations = []
    for comb in combinations:
        if in_square(*comb):
            in_square_combinations.append(comb)
        else:
            out_square_combinations.append(comb)
    size = get_size_str(len(combinations))
    in_square_size = get_size_str(len(in_square_combinations))
    out_square_size = get_size_str(len(out_square_combinations))
    print(f'{size=}, {in_square_size=}, {out_square_size=}')

    if not args.eval:
        HDF5_SIZE=1000
        if args.n_train >= 3:
            args.data_dir = data_dir / f'square_out_n={args.n_train}_{out_square_size}'
        else:
            args.data_dir = data_dir / f'square_out_{args.left}-{args.right}_{out_square_size}'
    else:
        HDF5_SIZE=100
        if not args.extrapolate:
            args.data_dir = data_dir / f'square_out_eval_{size}'
        else:
            args.data_dir = data_dir / f'square_out_eval_extrapolate_{size}'
    print(args.data_dir)
    
    # exit()
    
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
        
    if not args.eval:
        # for training data. only square_out data are included.
        if args.num_workers == 1:
            main_single_process(out_square_combinations) # for debug
        else:
            main(out_square_combinations)

    else:
        # for eval data. both square_in and square_out data are included.
        if args.num_workers == 1:
            main_single_process(combinations) # for debug
        else:
            main(combinations)

    merge_files(args.data_dir)

    # elif 'out_dist' in args.data_name:
    #     # ood eval data with different levels
    #     if 'xl' in args.data_name:
    #         ood_levels = out_dist_generate_xl()
    #     if 'cross' in args.data_name:
    #         ood_levels = out_dist_generate_cross()
    #     else:
    #         ood_levels = out_dist_generate()
    #     # print(ood_levels)
    #     for combinations, level in ood_levels:
    #         size = get_size_str(len(combinations))
    #         args.data_dir = pathlib.Path(f'/mnt/bn/magic/simple_scenes_data/collision_{size}_{args.data_name}_{level}')
    #         print(args.data_dir)
    #         if not os.path.exists(args.data_dir):
    #             os.mkdir(args.data_dir)
    #         if args.num_workers == 1:
    #             main_single_process(combinations) # for debug
    #         else:
    #             main(combinations)
    # else:
    #     raise NotImplementedError(args.data_name)

# orginal
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64

# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --left 1.25 --right 3.75
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --left 1.5 --right 3.5
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --left 1.75 --right 3.25
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --left 2 --right 3

# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --n_train 3
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 64 --n_train 4



# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 5 --eval
# python3 one_ball_uniform_motion_square_out_v0.py --num_workers 8 --eval --extrapolate