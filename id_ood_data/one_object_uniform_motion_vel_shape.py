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
def vectorized_draw_balls(positions, radii, color_name, object_type, width=512, height=512):
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
    
    colors = np.array([(255, 0, 0), (0, 0, 255)])
    if color_name == 'red':
        color = colors[0]
    elif color_name == 'blue':
        color = colors[1]
    else:
        raise NotImplementError
    

    n_frames = positions.shape[0]
    frames = np.ones((n_frames, height, width, 3), dtype=np.uint8) * 255

    # Adjust positions and radii for aspect ratio and normalized coordinate system
    norm_positions = positions
    norm_radii = radii

    if object_type == 'rectangle':
        half_w, half_h = radii, radii
        left = norm_positions[:, 0, None, None] - half_w
        right = norm_positions[:, 0, None, None] + half_w
        top = norm_positions[:, 1, None, None] - half_h
        bottom = norm_positions[:, 1, None, None] + half_h

        X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        mask = (X >= left) & (X <= right) & (Y >= top) & (Y <= bottom)
        frames[mask] = color

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
        frames = (1 - soft_mask) * frames + soft_mask * color

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





def simulate_one_ball_uniform_motion_in_one_dim(
    v1, r1, color, object_type,
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

    frames = vectorized_draw_balls(normed_positions, normed_radii, color, object_type, width=img_width, height=img_height)
    frames = frames[:, ::-1, :, :]

    return frames, positions, True

def generate_videos_hdf5(scenes, scenes_id, data_dir='./', seed=42, store_video=False):
    # fix randomness
    set_seeds(args.seed)

    batched_images = []
    batched_positions = []
    batched_init = []

    for scene in scenes:
        r1, v1, color, object_type = scene
        frames, positions, success = simulate_one_ball_uniform_motion_in_one_dim(v1, r1, color, object_type, world_scale=WORLD_SCALE, stride=STRIDE, numsteps=STRIDE*NUM_FRAMES)
        if not success: continue
        batched_images.append(frames)
        batched_positions.append(positions)
        
        if color == 'red':
            color_value = 0
        elif color == 'blue':
            color_value = 1
        else:
            raise NotImplementError
        
        if object_type == 'circle':
            object_value = 0
        elif object_type == 'rectangle':
            object_value = 1
        else:
            raise NotImplementError
        
        batched_init.append([r1, v1, color_value, object_value])

        if store_video or args.data_for_vis:
            video_dir = Path(data_dir) / 'vis_videos'
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            mp4_path = video_dir / f'{r1:.3f}_{v1:.3f}_{color}_{object_type}.mp4'
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

def generate_combinations(r1_list, v1_list, color, object_type):
    # Generate all possible combinations of r1, r2, v1, and v2
    combinations = []
    for r1 in r1_list:
        for v1 in v1_list:
            # Append the combination as a tuple to the list
            combinations.append((r1, v1, color, object_type))
            
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
    elif args.data_for_vis:
        V_SPACE= 4
        R_SPACE = 4
    low_v_list = np.linspace(MIN_V, MID_V, V_SPACE//2)
    high_v_list = np.linspace(MID_V, MAX_V, V_SPACE//2)
    r_list = np.linspace(MIN_R, MAX_R, R_SPACE)

    # List to hold all combinations
    combinations1 = generate_combinations(r_list, low_v_list, color='red', object_type='circle')
    combinations2 = generate_combinations(r_list, high_v_list, color='red', object_type='rectangle')
    combinations = combinations1 + combinations2
    np.random.shuffle(combinations) # in-place return None

    return combinations

def eval_dist_generate():
    # training or in-distribution eval
    print(f"args.data_name: {args.data_name}")
    
    V_SPACE= 31
    R_SPACE = 11
    
    low_v_list = np.linspace(MIN_V, MID_V, V_SPACE//2+1)
    high_v_list = np.linspace(MID_V, MAX_V, V_SPACE//2+1)
    r_list = np.linspace(MIN_R, MAX_R, R_SPACE)

    # List to hold all combinations
    combinations1 = generate_combinations(r_list, low_v_list, color='red', object_type='circle')
    combinations2 = generate_combinations(r_list, high_v_list, color='red', object_type='rectangle')
    combinations3 = generate_combinations(r_list, low_v_list, color='red', object_type='rectangle')
    combinations4 = generate_combinations(r_list, high_v_list, color='red', object_type='circle')
    combinations = combinations1 + combinations2 + combinations3 + combinations4
    np.random.shuffle(combinations) # in-place return None

    return combinations


def out_dist_generate():

    if not args.data_for_vis:
        v_id_list = np.linspace(MIN_V, MAX_V, 16)[::2]
        r_id_list = np.linspace(MIN_R, MAX_R, 6)
    else:
        # v_id_list = np.linspace(MIN_V, MAX_V, 4)
        # r_id_list = np.linspace(MIN_R, MAX_R, 4)
        v_id_list = np.linspace(MIN_V, MAX_V, 8)
        r_id_list = np.linspace(MIN_R, MAX_R, 8)

    MIN_OOD_V = 0.0
    MAX_OOD_V = 0.8
    MIN_OOD_R = 0.3
    MAX_OOD_R = 0.6

    if not args.data_for_vis:
        v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 11)
        r_ood_list =np.linspace(MIN_OOD_R, MAX_OOD_R, 11)
    else:
        # v_ood_list = np.array([0, 0.5, 0.8])
        # r_ood_list =np.array([0.35, 0.6])
        v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 16)
        r_ood_list = np.linspace(MIN_OOD_R, MAX_OOD_R, 8)

    # print(v_id_list, r_id_list, v_ood_list, r_ood_list)

    # Level 1: Only r1 is OOD; others are in distribution.
    combinations_level_1 = generate_combinations(r_ood_list, v_id_list)

    # Level 2: Only v1 is OOD; others are in distribution.
    combinations_level_2 = generate_combinations(r_id_list, v_ood_list)

    # Level 3: Both r1 and r2 are OOD.
    combinations_level_3 = generate_combinations(r_ood_list, v_ood_list)


    return (
        (combinations_level_1, 'level1_r1'),
        (combinations_level_2, 'level2_v1'),
        (combinations_level_3, 'level3_r1_r2'),
    )

def out_dist_generate_xl():

    if not args.data_for_vis:
        v_id_list = np.linspace(MIN_V, MAX_V, 16)[::2]
        r_id_list = np.linspace(MIN_R, MAX_R, 6)
    else:
        # v_id_list = np.linspace(MIN_V, MAX_V, 4)
        # r_id_list = np.linspace(MIN_R, MAX_R, 4)
        v_id_list = np.linspace(MIN_V, MAX_V, 16)
        r_id_list = np.linspace(MIN_R, MAX_R, 10)

    MIN_OOD_V = 4.5
    MAX_OOD_V = 6.0

    MIN_OOD_R = 1.5
    MAX_OOD_R = 2.0

    if not args.data_for_vis:
        v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 6)
        r_ood_list =np.linspace(MIN_OOD_R, MAX_OOD_R, 6)
    else:
        v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 16)
        r_ood_list =np.linspace(MIN_OOD_R, MAX_OOD_R, 8)
    # print(v_id_list, r_id_list, v_ood_list, r_ood_list)

    # Level 1: Only r1 is OOD; others are in distribution.
    combinations_level_1 = generate_combinations(r_ood_list, v_id_list)

    # Level 2: Only v1 is OOD; others are in distribution.
    combinations_level_2 = generate_combinations(r_id_list, v_ood_list)

    # Level 3: Both r1 and r2 are OOD.
    combinations_level_3 = generate_combinations(r_ood_list, v_ood_list)

    return (
        (combinations_level_1, 'xl_level1_r1'),
        (combinations_level_2, 'xl_level2_v1'),
        (combinations_level_3, 'xl_level3_r1_r2'),
    )


def out_dist_generate_cross():

    if not args.data_for_vis:
        v_id_list = np.linspace(MIN_V, MAX_V, 16)[::2]
        r_id_list = np.linspace(MIN_R, MAX_R, 6)
    else:
        # v_id_list = np.linspace(MIN_V, MAX_V, 4)
        # r_id_list = np.linspace(MIN_R, MAX_R, 4)
        v_id_list = np.linspace(MIN_V, MAX_V, 8)
        r_id_list = np.linspace(MIN_R, MAX_R, 8)


    MIN_OOD_V = 0.0
    MAX_OOD_V = 0.8
    MIN_OOD_R = 0.3
    MAX_OOD_R = 0.6

    if not args.data_for_vis:
        left_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 6)
        left_r_ood_list =np.linspace(MIN_OOD_R, MAX_OOD_R, 6)
    else:
        # left_v_ood_list = np.array([0, 0.5, 0.8])
        # left_r_ood_list =np.array([0.35, 0.6])
        left_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 16)
        left_r_ood_list = np.linspace(MIN_OOD_R, MAX_OOD_R, 8)

    MIN_OOD_V = 4.5
    MAX_OOD_V = 6.0

    MIN_OOD_R = 1.5
    MAX_OOD_R = 2.0

    if not args.data_for_vis:
        right_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 6)
        right_r_ood_list =np.linspace(MIN_OOD_R, MAX_OOD_R, 6)
    else:
        # right_v_ood_list = np.array([4.5, 6.0])
        # right_r_ood_list = np.array([1.5, 2.0])
        right_v_ood_list = np.linspace(MIN_OOD_V, MAX_OOD_V, 16)
        right_r_ood_list = np.linspace(MIN_OOD_R, MAX_OOD_R, 8)



    # print(v_id_list, r_id_list, v_ood_list, r_ood_list)

    # Level 1: Only r1 is OOD; others are in distribution.
    # combinations_level_1 = generate_combinations(r_ood_list, r_id_list, v_id_list, v_id_list)

    # Level 2: Only v1 is OOD; others are in distribution.
    # combinations_level_2 = generate_combinations(r_id_list, r_id_list, v_ood_list, v_id_list)

    # Level 3: Both r1 and r2 are OOD.
    # combinations_level_3 = generate_combinations(left_r_ood_list, right_r_ood_list, v_id_list, v_id_list)

    # Level 4: Both v1 and v2 are OOD.
    # combinations_level_4 = generate_combinations(r_id_list, r_id_list, left_v_ood_list, right_v_ood_list)

    # Level 5: v1 and r1 are OOD; v2 and r2 are in distribution.
    # combinations_level_5 = generate_combinations(r_ood_list, r_id_list, v_ood_list, v_id_list)

    # Level 6: All (v1, v2, r1, r2) are OOD.
    combinations_level_6 = generate_combinations(left_r_ood_list, right_v_ood_list)
    combinations_level_7 = generate_combinations(right_r_ood_list, left_v_ood_list)

    return (
        # (combinations_level_1, 'xl_level1_r1'),
        # (combinations_level_2, 'xl_level2_v1'),
        # (combinations_level_3, 'cross_level3_r1_r2'),
        # (combinations_level_4, 'cross_level4_v1_v2'),
        # (combinations_level_5, 'xl_level5_r1_v1'),
        (combinations_level_6, 'cross_level6_r1_r2_v1_v2'),
        (combinations_level_7, 'cross_level6_r2_r1_v1_v2'),
    )


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
    # parser.add_argument('--data_ood_level', type=int) # data ood level for ood eval
    parser.add_argument('--data_name', type=str, default='v1')
    parser.add_argument('--data_for_vis', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
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
    MID_V = 2.5
    
    MIN_R = 0.7
    MAX_R = 1.4
    MID_R = 1.05
    DIRECTION = 'right'
    

    data_dir = pathlib.Path(f'/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion')

    if not args.eval:
        # training data
        combinations = in_dist_generate()
        size = get_size_str(len(combinations))
        args.data_dir = data_dir / f'vel_shape_{size}'
        HDF5_SIZE=1000
        
    else:
        # eval data
        combinations = eval_dist_generate()
        size = get_size_str(len(combinations))
        args.data_dir = data_dir / f'vel_shape_eval_{size}'
        HDF5_SIZE=100
        
        
    print(args.data_dir)
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if args.num_workers == 1:
        main_single_process(combinations) # for debug
    else:
        main(combinations)
            
    merge_files(args.data_dir)




# python3 one_object_uniform_motion_vel_shape.py --data_name in_dist_v2 --data_for_vis --num_workers 1
# python3 one_object_uniform_motion_vel_shape.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64
# python3 one_object_uniform_motion_vel_shape.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64 --eval