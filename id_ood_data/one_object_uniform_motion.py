import numpy as np
import time
import cv2
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
from PIL import Image

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


def uniform_random(low, high):
    return np.random.rand() * (high - low) + low


def vectorized_draw_objects(positions, radii, colors=[(255, 0, 0), (0, 0, 255)], width=512, height=512):
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
    
    object_type = args.object_type
    n_frames = positions.shape[0]
    frames = np.ones((n_frames, height, width, 3), dtype=np.uint8) * 255
    colors = np.array(colors)

    # Adjust positions and radii for aspect ratio and normalized coordinate system
    norm_positions = positions
    norm_radii = radii

    if object_type == 'rectangle':
        half_w, half_h = radii
        left = norm_positions[:, 0, None, None] - half_w
        right = norm_positions[:, 0, None, None] + half_w
        top = norm_positions[:, 1, None, None] - half_h
        bottom = norm_positions[:, 1, None, None] + half_h

        X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        mask = (X >= left) & (X <= right) & (Y >= top) & (Y <= bottom)
        frames[mask] = colors[0]

    elif object_type == 'circle':
        # Prepare meshgrid for coordinates, normalize to [0, 1] range
        X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))


        # Calculate squared distances for each ball in normalized coordinates
        dx = X - norm_positions[:, 0, np.newaxis, np.newaxis]
        dy = Y - norm_positions[:, 1, np.newaxis, np.newaxis]
        
        distance_squared = dx**2 + dy**2
        radius_squared = norm_radii[0]**2
        # Calculate a soft mask based on distance to the circle's edge
        edge_width = 0.03  # Control the width of the edge transition
        soft_mask = np.clip((radius_squared - distance_squared) / (radius_squared * edge_width), 0, 1)[:,:,:,None]
        
        # print(soft_mask.shape, frames.shape, colors[i].shape)
        frames = (1 - soft_mask) * frames + soft_mask * colors[0]

    elif object_type == 'car':
        car_icon = Image.open('icons/car.jpeg').convert('RGB')
        car_icon = np.array(car_icon)
        half_w, half_h = int(round(radii[0]*width)), int(round(radii[1]*height))
        # print(car_icon.shape) # (185, 389, 3)
        # car_icon = np.resize(car_icon, (half_h*2, half_w*2, 3))
        car_icon = cv2.resize(car_icon, dsize=(half_w*2, half_h*2), interpolation=cv2.INTER_CUBIC)
        car_icon = np.array(car_icon)[::-1]

        # the coordinate of the center of the car icon
        x = (positions[:, 0] * width).astype(int)
        y = (positions[:, 1] * height).astype(int)

        # check not out of bound
        # assert (x >= 0).all() and (x < width).all() and (y >= 0).all() and (y < height).all()

        # the bound of the car
        x1 = np.clip(x - half_w, 0, width)
        x2 = np.clip(x + half_w, 0, width)
        y1 = np.maximum(0, y - half_h)
        y2 = np.minimum(height, y + half_h)

        car_x1 = half_w - (x - x1)
        car_x2 = half_w + (x2 - x)
        car_y1 = half_h - (y - y1)
        car_y2 = half_h + (y2 - y)

        for i in range(n_frames):
            # print(x1[i], x2[i], y1[i], y2[i])
            # print(car_x1[i], car_x2[i], car_y1[i], car_y2[i])
            # print(car_icon.shape)
            # print(frames[i, y1[i]:y2[i], x1[i]:x2[i]].shape)
            # print(car_icon[car_y1[i]:car_y2[i], car_x1[i]:car_x2[i]].shape)
            frames[i, y1[i]:y2[i], x1[i]:x2[i]] = car_icon[car_y1[i]:car_y2[i], car_x1[i]:car_x2[i]]

    return frames.astype(np.uint8)


def simulate_one_object_uniform_motion_in_one_dim(
    v1, r1,
    world_scale,
    x1=None, ball_height=None, 
    img_width=256, img_height=256,
    timestep=0.01, numsteps=500, stride=10,
    eps = 0.05,
    ):

    if x1 is None: 
        # ensure we have at least MIN_PRE_FRAMES frames before out of vision
        min_dist = NUM_MIN_FRAMES * v1 * timestep * stride + r1[0]
        max_init = world_scale - min_dist
        if max_init < r1[0]:
            # print(f"{max_init=}, {r1=}, {v1=}")
            max_init = r1[0] + uniform_random(0, 0.5)
        x1 = uniform_random(r1[0], max_init)

    if ball_height is None: 
        max_r = r1[1]
        ball_height = uniform_random(
            max_r + eps * world_scale,
            (1 - eps) * world_scale - max_r,
        )

    world = b2World(gravity=(0, 0), doSleep=False)

    if DIRECTION == 'right':
        ball1 = world.CreateDynamicBody(position=(x1, ball_height), linearVelocity=(v1, 0))
        ball1.CreateCircleFixture(radius=r1[0], density=1, restitution=1.0)
    elif DIRECTION == 'left':
        ball1 = world.CreateDynamicBody(position=(world_scale - x1, ball_height), linearVelocity=(-v1, 0))
        ball1.CreateCircleFixture(radius=r1[0], density=1, restitution=1.0)
    elif DIRECTION == 'upper':
        ball1 = world.CreateDynamicBody(position=(ball_height, x1), linearVelocity=(0, v1))
        ball1.CreateCircleFixture(radius=r1[0], density=1, restitution=1.0)
    elif DIRECTION == 'down':
        ball1 = world.CreateDynamicBody(position=(ball_height, world_scale - x1), linearVelocity=(0, -v1))
        ball1.CreateCircleFixture(radius=r1[0], density=1, restitution=1.0)

    ball1_xy = []
    for i in range(numsteps):
        world.Step(timestep, 15, 20)
        if i % stride == 0:
            ball1_xy.append(np.array(ball1.position))

    positions = np.array(ball1_xy)
    radii = np.array(r1)
    # print(positions)

    # normlazied
    normed_positions = positions / world_scale
    normed_radii = radii / world_scale

    frames = vectorized_draw_objects(normed_positions, normed_radii, width=img_width, height=img_height)
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
        frames, positions, success = simulate_one_object_uniform_motion_in_one_dim(v1, r1, world_scale=WORLD_SCALE, stride=STRIDE, numsteps=STRIDE*NUM_FRAMES)
        if not success: continue
        batched_images.append(frames)
        batched_positions.append(positions)
        batched_init.append([r1[0], r1[1], v1])

        if store_video:
            video_dir = Path(data_dir) / 'vis_videos'
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            mp4_path = video_dir / f'{r1[0]:.3f}_{r1[1]:.3f}_{v1:.3f}.mp4'
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
    for r in r1_list:   
        for v1 in v1_list:
            # Append the combination as a tuple to the list
            combinations.append(((r, r), v1))
            
    np.random.shuffle(combinations) # in-place return None
    return combinations

def in_dist_generate():
    # training or in-distribution eval
    print(f"args.data_name: {args.data_name}")
    
    if args.data_size_level == 0:
        V_SPACE= 300
        R_SPACE = 100
    elif args.data_size_level == 1:
        V_SPACE= 1000
        R_SPACE = 300
    elif args.data_size_level == 2:
        V_SPACE= 3000
        R_SPACE = 1000
    else:
        V_SPACE= 10
        R_SPACE = 10
    v_list = np.linspace(MIN_V, MAX_V, V_SPACE)
    r_list = np.linspace(MIN_R, MAX_R, R_SPACE)

    if not args.data_for_vis:
        # V_SPACE= 15
        v_list = sorted([uniform_random(MIN_V, MAX_V) for _ in range(V_SPACE)])
        r_list = sorted([uniform_random(MIN_R, MAX_R) for _ in range(R_SPACE)])
    else:
        v_list = np.linspace(MIN_V, MAX_V, 10)
        r_list = np.linspace(MIN_R, MAX_R, 10)


    # List to hold all combinations
    combinations = generate_combinations(r_list, v_list)

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
    parser.add_argument('--data_size_level', type=int) # data size level for training
    parser.add_argument('--data_name', type=str, default='v1')
    parser.add_argument('--object_type', type=str, default='rectangle')
    parser.add_argument('--data_for_vis', default=False, action='store_true')
    args = parser.parse_args()
    
    # fix randomness
    set_seeds(args.seed)
        
    WORLD_SCALE = 10.0 # the scale of imagined world. Set positions and velocitys by this scale
    STRIDE = 10
    NUM_MIN_FRAMES = 24
    NUM_FRAMES = 32
    HDF5_SIZE=1000
    FPS=5

    MIN_V = 1.0
    MAX_V = 4.0
    if args.object_type in ['circle', 'ball']:
        MIN_R = 0.3
        MAX_R = 0.6
    elif args.object_type == 'rectangle':
        MIN_R = 0.7
        MAX_R = 1.4
    elif args.object_type == 'car':
        MIN_R = 1.5
        MAX_R = 2.0
    else:
        raise NotImplementedError
    DIRECTION = 'right'

    data_dir = pathlib.Path(f'./moving_objects')
    if args.data_for_vis:
        data_name = f'vis_data_{args.object_type}_L'
        for direction in ['left', 'upper', 'down']:
            if direction in args.data_name:
                data_name = f'vis_data_{args.object_type}_{direction}_L'
                DIRECTION = direction

        # train and eval data for fast eval
        args.data_name = 'eval'
        args.data_dir = data_dir/data_name
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)
        combinations = in_dist_generate()
        print(f'number of samples for visualization is {len(combinations)}')
        main_vis(combinations)

    elif 'eval' not in args.data_name or 'in_dist' in args.data_name:
        # training data
        combinations = in_dist_generate()
        size = get_size_str(len(combinations))
        args.data_dir = data_dir / f'data_{size}_{args.object_type}'
        if not os.path.exists(args.data_dir):
            os.mkdir(args.data_dir)
        if args.num_workers == 1:
            main_single_process(combinations) # for debug
        else:
            main(combinations)

    merge_files(args.data_dir)

# python3  one_object_uniform_motion.py  --data_for_vis  --object_type rectangle
# python3  one_object_uniform_motion.py  --data_for_vis  --object_type car
# python3  one_object_uniform_motion.py  --data_for_vis  --object_type circle


# python3  one_object_uniform_motion.py  --data_for_vis --data_name right
# sudo /mnt/bn/magic/yueyang/miniconda3/envs/phyre/bin/python3 one_object_uniform_motion.py --data_name in_dist_v2 --data_size_level 0 --num_workers 64
# sudo /mnt/bn/magic/yueyang/miniconda3/envs/phyre/bin/python3 one_object_uniform_motion.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64
# sudo /mnt/bn/magic/yueyang/miniconda3/envs/phyre/bin/python3 one_object_uniform_motion.py --data_name in_dist_v2 --data_size_level 2 --num_workers 64