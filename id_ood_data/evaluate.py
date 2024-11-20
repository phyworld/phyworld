import cv2
import numpy as np
from pathlib import Path
import imageio
import os
from random import sample
import h5py
import numpy as np
import imageio
import tempfile
import imageio.v3 as iio # need python at least 3.9
from pathlib import Path
import time
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import pandas as pd


color = (255, 0, 0)
left_color = (255, 0, 0)
right_color = (0, 0, 255)
COLORS = [left_color, right_color]

WORLD_SCALE = 10.0



# Now we need a function to find objects of a particular color
# def find_objects_of_color(image_rgb, color, color_tol=120):
#     # Convert color and color_tol to int to prevent overflow
#     color = np.array(color, dtype=np.int32)

#     # Calculate min and max color values
#     min_color = np.clip(color - color_tol, 0, 255)
#     max_color = np.clip(color + color_tol, 0, 255)

#     # Convert back to uint8
#     min_color = np.array(min_color, dtype=np.uint8)
#     max_color = np.array(max_color, dtype=np.uint8)
#     # print(min_color, max_color)
#     mask = cv2.inRange(image_rgb, min_color, max_color)
#     return (mask == 255).astype(np.uint8)

# Helper function to find circles with Hough Transform
def find_circles(mask, dp, minDist, param1, param2, minRadius, maxRadius):
    return cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp, minDist,
                            param1=param1, param2=param2,
                            minRadius=minRadius, maxRadius=maxRadius)

def is_valid_circle(circle, color_mask):
    # Create a mask for the circle
    circle_img = np.zeros_like(color_mask)
    cv2.circle(circle_img, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 255, 255), -1)

    # Perform bitwise AND to find the colored area within the circle
    colored_area = cv2.bitwise_and(color_mask, color_mask, mask=circle_img)

    # Calculate the area ratio
    circle_area = np.pi * (circle[2] ** 2)
    colored_area_count = np.count_nonzero(colored_area)
    area_ratio = colored_area_count / circle_area

    # Set a threshold for the minimum area ratio for a circle to be considered valid
    area_ratio_threshold = 0.85  # This threshold may need to be adjusted
    # print(area_ratio)

    return area_ratio > area_ratio_threshold

def filter_circles(circles, color_mask):
    # Sort the circles based on their radius, largest first
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    
    # Filter out circles based on the area ratio of colored pixels
    circles = [circle for circle in circles if is_valid_circle( circle, color_mask)]
    
    filtered_circles = []
    
    for circle in circles:
        center, radius = circle[:2], circle[2]
        # # Check if this circle is inside any already accepted larger circle
        if not any(np.linalg.norm(np.array(center) - np.array(other_center)) < other_radius 
                   for other_center, other_radius in filtered_circles):
            filtered_circles.append((center, radius))
    
    return filtered_circles

def fill_missing_values(list_a, list_b):
    # Make sure all elements are numpy arrays
    
    # Initialize the result list with NaN vectors of the same dimension as list_b's elements
    result = [np.full_like(list_b[0], np.nan) for _ in range(len(list_b))]
    
    # Iterate through each element in list_a
    for idx_a, elem_a in enumerate(list_a):
        # Calculate the distance between elem_a and each element in list_b
        distances = [np.linalg.norm(elem_a - elem_b) for elem_b in list_b]
        # Find the index of the closest element in list_b
        idx_b = np.argmin(distances)
        # Place elem_a in the result list at the position of the closest element
        result[idx_b] = elem_a
    
    return result

def parse_state_from_image(image_rgb,  default_y, default_r1, thres=0.15, color=0):
    # Copy of the image for drawing
    image_copy = image_rgb.copy()

    all_scaled_circles = []

    # Go through all the colors in the WAD_COLORS array except for black and white
    # For circles
    # circle_mask = find_objects_of_color(image_copy, color)
    if color == 0:
        circle_mask = (image_copy[:, :, 0] > 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] < 127)
    elif color == 1:
        circle_mask = (image_copy[:, :, 0] < 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] > 127)
    else:
        raise ValueError("Invalid color")

    # print(circle_mask.shape, circle_mask.min(), circle_mask.max(), np.unique(circle_mask))
    # plt.imshow(circle_mask)
    # plt.axis('off') # Hide axis
    # plt.show()

    area = np.sum(circle_mask)
    radius = np.sqrt(area / np.pi)
    scaled_radius = radius / image_copy.shape[1] * WORLD_SCALE
    # print(area)
    if area > thres:
        center = (
            # int(np.mean(np.nonzero(circle_mask)[1])),
            # int(np.mean(np.nonzero(circle_mask)[0])),
            np.mean(np.nonzero(circle_mask)[1]),
            np.mean(np.nonzero(circle_mask)[0]),
                )
        scaled_center = (
                center[0] / image_copy.shape[1] * WORLD_SCALE,
                center[1] / image_copy.shape[1] * WORLD_SCALE,
            )
    else:
        # raidus, scaled_radius = np.nan,  np.nan # represent non-existance
        # center = (np.nan, np.nan)
        # scaled_center = (np.nan, np.nan)
        scaled_center = (WORLD_SCALE, default_y)
        scaled_radius = default_r1
            

    all_scaled_circles.append([*scaled_center, scaled_radius])

    # Draw the filtered circles
    # for center, radius in all_circles:
    #     cv2.circle(image_copy, center, int(radius), 
    #                (0, 0, 0), 1)
    # plt.imshow(image_copy)
    # plt.axis('off') # Hide axis
    # plt.show()
    
    return np.array(all_scaled_circles)


def parse_state_from_image_collision(image_rgb,  default_y, default_r1, default_r2, thres=0.15, ):
    # Copy of the image for drawing
    image_copy = image_rgb.copy()

    # Store all circles in a list
    all_scaled_circles = []
    # print(circles[0])

    # Go through all the colors in the WAD_COLORS array except for black and white
    for ball_id, color in enumerate(COLORS):  # Skipping white
        # For circles
        # circle_mask = find_objects_of_color(image_copy, color)
        if color == (255, 0, 0):
            circle_mask = (image_copy[:, :, 0] > 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] < 127)
        elif color == (0, 0, 255):
            circle_mask = (image_copy[:, :, 0] < 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] > 127)
        else:
            raise ValueError("Invalid color")

        # print(circle_mask.shape, circle_mask.min(), circle_mask.max(), np.unique(circle_mask))
        # plt.imshow(circle_mask)
        # plt.axis('off') # Hide axis
        # plt.show()

        area = np.sum(circle_mask)
        radius = np.sqrt(area / np.pi)
        scaled_radius = radius / image_copy.shape[1] * WORLD_SCALE
        if area > thres:
            center = (
                # int(np.mean(np.nonzero(circle_mask)[1])),
                # int(np.mean(np.nonzero(circle_mask)[0])),
                np.mean(np.nonzero(circle_mask)[1]),
                np.mean(np.nonzero(circle_mask)[0]),
                 )
            scaled_center = (
                    center[0] / image_copy.shape[1] * WORLD_SCALE,
                    center[1] / image_copy.shape[1] * WORLD_SCALE,
                )
        else:
            # raidus, scaled_radius = np.nan,  np.nan # represent non-existance
            # center = (np.nan, np.nan)
            # scaled_center = (np.nan, np.nan)
            if ball_id == 0:
                scaled_center = (0, default_y)
                scaled_radius = default_r1
                # scaled_radius = np.nan
            else:
                scaled_center = (WORLD_SCALE, default_y)
                scaled_radius = default_r2
                # scaled_radius = np.nan
                

        all_scaled_circles.append([*scaled_center, scaled_radius])

    # Draw the filtered circles
    # for center, radius in all_circles:
    #     cv2.circle(image_copy, center, int(radius), 
    #                (0, 0, 0), 1)
    # plt.imshow(image_copy)
    # plt.axis('off') # Hide axis
    # plt.show()
    
    return np.array(all_scaled_circles)

def get_last_ema(values_list, span):
    # Convert the list to a pandas Series
    series = pd.Series(values_list)
    # Calculate the EMA with specified span
    ema = series.ewm(span=span, adjust=False).mean()
    # Return the last EMA value
    return ema.iloc[-1]

def xy_metrics(list_a, list_b):
    distances_x, distances_y = [], []
    # print(f'{list_a=}, {list_b=}')
    assert len(list_a) == len(list_b) == 1
    for elem_a, elem_b in zip(list_a, list_b):
        # Check if elem_a is not NaN. Since elem_a is a vector, we need to check if any of its values are NaN.
        if not np.any(np.isnan(elem_a)) and not np.any(np.isnan(elem_b)):
            # Calculate the L2 distance for non-NaN elements
            distances_x.append(np.abs(elem_a[0] - elem_b[0]))
            distances_y.append(np.abs(elem_a[1] - elem_b[1]))
    # no ball is detected
    x_error_avg = np.mean(distances_x) if distances_x else np.nan
    y_error_avg = np.mean(distances_y) if distances_y else np.nan
    # normalized to [0, 100]
    # x_error_avg = x_error_avg / WORLD_SCALE * 100
    # y_error_avg = y_error_avg / WORLD_SCALE * 100
    return x_error_avg, y_error_avg

def xy_metrics_collision(list_a, list_b):
    distances_x, distances_y = [], []
    assert len(list_a) == len(list_b) == 2
    for elem_a, elem_b in zip(list_a, list_b):
        # Check if elem_a is not NaN. Since elem_a is a vector, we need to check if any of its values are NaN.
        if not np.any(np.isnan(elem_a)) and not np.any(np.isnan(elem_b)):
            # Calculate the L2 distance for non-NaN elements
            distances_x.append(np.abs(elem_a[0] - elem_b[0]))
            distances_y.append(np.abs(elem_a[1] - elem_b[1]))
    # no ball is detected
    x_error_avg = np.mean(distances_x) if distances_x else np.nan
    y_error_avg = np.mean(distances_y) if distances_y else np.nan
    # normalized to [0, 100]
    # x_error_avg = x_error_avg / WORLD_SCALE * 100
    # y_error_avg = y_error_avg / WORLD_SCALE * 100
    return x_error_avg, y_error_avg



def evaluate_xy(rollout_frames, gt_features, init, mode, gamma=0.98, sample_freq=1, pred_states=None):
    assert sample_freq == 1, 'there may be some bugs if it is greater than 1'

    # print(f'{init.shape=}, {gt_features.shape=}')
    # cut off the lastframes without balls
    left_ball_r = init[0]
    left_ball_init_v = init[1]
    left_ball_m = init[0]**2
    index = []
    CONDITION_FRAMES = 4
    for i, state in enumerate(gt_features):
        if i < CONDITION_FRAMES-1:
            continue
        left_ball_x, left_ball_y = state[0], state[1]
        if left_ball_x - left_ball_r >= 0 and left_ball_x + left_ball_r <= WORLD_SCALE \
            and left_ball_y - left_ball_r >= 0 and left_ball_y + left_ball_r <= WORLD_SCALE:
            index.append(i)
    # print(left_ball_r, right_ball_r, left_ball_x, right_ball_x, i)
    # print(len(rollout_frames), '-->', len(index))


    if rollout_frames is not None:
        assert len(rollout_frames) == len(gt_features), f'{len(rollout_frames)}, {len(gt_features)}'
        rollout_frames = rollout_frames[index]
        gt_features = gt_features[index]

        x_error_list, y_error_list, r_list = [], [], []
        x_pos_list, y_pos_list = [], []
        default_y, default_r1 = np.nan, np.nan
        for rollout_frame, gt_feature in zip(rollout_frames[sample_freq-1::sample_freq], gt_features[sample_freq-1::sample_freq]):
            parsed_state = parse_state_from_image(rollout_frame, default_y, default_r1, color=init[2] if len(init) >=3 else 0)
            default_y, default_r1 = parsed_state[0][1], parsed_state[0][2]
            x_pos_list.append(parsed_state[:, 0])
            y_pos_list.append(parsed_state[:, 1])
            r_list.append(parsed_state[:, 2])
            parsed_state = parsed_state[:, :2]
            # print('gt: ', gt_feature)
            # print('rollout: ', parsed_state)
            x_error, y_error = xy_metrics([gt_feature], parsed_state)
            # print(x_error, y_error)
            x_error_list.append(x_error)
            y_error_list.append(y_error)
        span = len(rollout_frames[sample_freq-1::sample_freq])
        # print(x_error_list)
        # print(y_error_list)
        ema_x_error = get_last_ema(x_error_list, span=span)
        ema_y_error = get_last_ema(y_error_list, span=span)

    else:
        assert pred_states is not None and len(pred_states) == len(gt_features), f'{len(pred_states)}, {len(gt_features)}'

        # print(gt_features[:, :, 0])
        # print(pred_states[:, :2])

        pred_states = pred_states[index]
        gt_features = gt_features[index]  
        x_pos_list = pred_states[:, 0]
        y_pos_list = pred_states[:, 1]
        r_list =  pred_states[:, 2]


    # print('last frame', x_error_list[-1], y_error_list[-1])
    # print(r_list)

    # only use frames before collision to avoid out of vision region
    # MIN_POST_FRAMES = 8
    # r_list = r_list[:(MIN_POST_FRAMES-CONDITION_FRAMES+1)//sample_freq] 
    r1_list = r_list
    r1_min, r1_max = min(r1_list), max(r1_list)
    delta_r = r1_max - r1_min


    gt_x_pos = gt_features[:, 0]
    gt_x_vel = np.diff(gt_x_pos, axis=0) / (0.1 * sample_freq)
    gt_y_pos = gt_features[:, 1]
    gt_y_vel = np.diff(gt_y_pos, axis=0) / (0.1 * sample_freq)

    # calculate velocity of x
    x_pos = np.array(x_pos_list) # shape (timestep, 1, )
    y_pos = np.array(y_pos_list) # shape (timestep, 1, )
    if x_pos.shape[-1] == 1:
        x_pos = np.squeeze(x_pos, -1)
        y_pos = np.squeeze(y_pos, -1)

    x_vel = np.diff(x_pos, axis=0) / (0.1 * sample_freq) # shape (timestep-1, 2, )
    y_vel = np.diff(y_pos, axis=0) / (0.1 * sample_freq) # shape (timestep-1, 2, )

    x_err_avg = np.mean(np.abs(x_pos - gt_x_pos))
    y_err_avg = np.mean(np.abs(y_pos - gt_y_pos))

    # print(x_pos)
    # print(gt_x_pos)

    # print(gt_x_pos)
    # print(gt_x_pos[:collision_index+1], gt_x_pos[collision_index+2:])

    # print(x_vel.shape, gt_x_vel.shape)
    # print(x_vel, gt_x_vel)
    # print(gt_x_vel[:collision_index], gt_x_vel[collision_index+1:])
    x_vel_err_avg = np.mean(np.abs(x_vel - gt_x_vel)) 
    y_vel_err_avg = np.mean(np.abs(y_vel - gt_y_vel)) 
    # print(f'{post_vel_err_avg=}')

    scale = max(0.5, np.abs(left_ball_init_v))

    if mode == 'all':
        # return ema_x_error, ema_y_error, x_error_list[-1], y_error_list[-1], delta_r
        return {
            'init': init,
            # parsed data
            'x_pos': x_pos,
            'y_pos': y_pos,
            'x_vel': x_vel,
            'y_vel': y_vel,
            'r': r_list,
            'gt_x_pos': gt_x_pos,
            'gt_y_pos': gt_y_pos,
            # error
            'abs_x_err_avg': x_err_avg,
            'abs_y_err_avg': y_err_avg,
            'delta_r': delta_r, # reflect consistency of rollout frames. Not the error w.r.t. gt.
            'abs_x_vel_err_avg': x_vel_err_avg,
            'abs_y_vel_err_avg': y_vel_err_avg,
        }
    else:
        return ema_x_error, ema_y_error



def evaluate_xy_collision(rollout_frames, gt_features, init, mode, gamma=0.98, sample_freq=1, pred_states=None):
    assert sample_freq == 1, 'there may be some bugs if it is greater than 1'


    # cut off the lastframes without balls
    left_ball_r, right_ball_r = init[:2]
    left_ball_init_v, right_ball_init_v = init[2:4]
    left_ball_m, right_ball_m = init[:2]**2
    index = []
    CONDITION_FRAMES = 4
    for i, state in enumerate(gt_features):
        if i < CONDITION_FRAMES-1:
            continue
        left_ball_x = state[0, 0]
        right_ball_x = state[1, 0]
        if left_ball_x - left_ball_r >= 0 and right_ball_x + right_ball_r <= WORLD_SCALE:
            index.append(i)
    # print(left_ball_r, right_ball_r, left_ball_x, right_ball_x, i)
    # print(len(rollout_frames), '-->', len(index))
        
    if rollout_frames is not None:

        """set large sample_freq to save computational cost"""
        assert len(rollout_frames) == len(gt_features), f'{len(rollout_frames)}, {len(gt_features)}'

        rollout_frames = rollout_frames[index]
        gt_features = gt_features[index]

        r_list = []
        x_pos_list = []
        default_y, default_r1, default_r2 = np.nan, np.nan, np.nan
        for rollout_frame, gt_feature in zip(rollout_frames[sample_freq-1::sample_freq], gt_features[sample_freq-1::sample_freq]):
            parsed_state = parse_state_from_image_collision(rollout_frame, default_y, default_r1, default_r2)
            default_y, default_r1, default_r2 = parsed_state[0][1], parsed_state[0][2], parsed_state[1][2]
            x_pos_list.append(parsed_state[:, 0])
            r_list.append(parsed_state[:, 2])
            parsed_state = parsed_state[:, :2]
    
    else:
        assert pred_states is not None and len(pred_states) == len(gt_features), f'{len(pred_states)}, {len(gt_features)}'

        # print(gt_features[:, :, 0])
        # print(pred_states[:, :2])

        pred_states = pred_states[index]
        gt_features = gt_features[index]  
        x_pos_list = pred_states[:, :2]
        r_list =  pred_states[:, 4:]
        
    # only use frames before collision to avoid out of vision region
    MIN_POST_FRAMES = 8
    r_list = r_list[:(MIN_POST_FRAMES-CONDITION_FRAMES+1)//sample_freq] 
    r1_list = [x[0] for x in r_list]
    r2_list = [x[1] for x in r_list]
    r1_min, r1_max = min(r1_list), max(r1_list)
    r2_min, r2_max = min(r2_list), max(r2_list)
    delta_r = max(r1_max - r1_min, r2_max - r2_min)


    gt_x_pos = gt_features[:, :, 0]
    gt_x_vel = np.diff(gt_x_pos, axis=0) / (0.1 * sample_freq)

    # find when collision happens by gt_x_pos (at least one of balls change the trend of its x_pos)
    collision_index = len(gt_x_vel) - 1 # if no frames after collision
    for i in range(1, len(gt_x_vel)):
        # if gt_x_pos[i, 0] - gt_x_pos[i-1, 0] <= 0 or gt_x_pos[i, 1] - gt_x_pos[i-1, 1] >= 0:
        # if the value of gt_x_vel changes greatly
        if np.abs(gt_x_vel[i, 0] - gt_x_vel[i-1, 0]) > 0.1 or np.abs(gt_x_vel[i, 1] - gt_x_vel[i-1, 1]) > 0.1:
            # print(f'{i=}, {gt_x_vel[i, 0]=}, {gt_x_vel[i, 1]=}, {gt_x_vel[i-
            collision_index = i
            break

    # calculate velocity of x
    x_pos = np.array(x_pos_list) # shape (timestep, 2, )
    x_vel = np.diff(x_pos, axis=0) / (0.1 * sample_freq) # shape (timestep-1, 2, )

    pre_x_err_avg = np.mean(np.abs(x_pos[:collision_index+1] - gt_x_pos[:collision_index+1])) 
    post_x_err_avg = np.mean(np.abs(x_pos[collision_index+1:] - gt_x_pos[collision_index+1:])) 

    pre_vel_err_avg = np.mean(np.abs(x_vel[:collision_index] - gt_x_vel[:collision_index])) 
    post_vel_err_avg = np.mean(np.abs(x_vel[collision_index+1:] - gt_x_vel[collision_index+1:])) 

    momentum = left_ball_m *  x_vel[:, 0] + right_ball_m * x_vel[:, 1]
    gt_momentum = left_ball_m *  left_ball_init_v - right_ball_m * right_ball_init_v
    # print(f'{gt_momentum=}, {momentum=}')
    # print(momentum[:collision_index], momentum[collision_index+1:])
    pre_momentum_error_avg = np.mean(np.abs(momentum[:collision_index] - gt_momentum))
    post_momentum_error_avg = np.mean(np.abs(momentum[collision_index+1:] - gt_momentum))

    # calculate energy
    energy = left_ball_m * x_vel[:, 0]**2 / 2 + right_ball_m * x_vel[:, 1]**2 / 2
    gt_energy = left_ball_m * left_ball_init_v**2 / 2 + right_ball_m * right_ball_init_v**2 / 2
    pre_energy_error_avg = np.mean(np.abs(energy[:collision_index] - gt_energy))
    post_energy_error_avg = np.mean(np.abs(energy[collision_index+1:] - gt_energy))


    scale = max(0.5, np.abs(left_ball_init_v))

    # return ema_x_error, ema_y_error, x_error_list[-1], y_error_list[-1], delta_r
    return {
        'init': init,
        # 
        'pre_x_vel': x_vel[:collision_index],
        'gt_pre_x_vel': gt_x_vel[:collision_index],
        'post_x_vel': x_vel[collision_index+1:],
        'gt_post_x_vel': gt_x_vel[collision_index+1:],
        # 'x_error_avg': ema_x_error,
        # 'y_error_avg': ema_y_error,
        # 'x_error_last': x_error_list[-1],
        # 'y_error_last': y_error_list[-1],
        'delta_r': delta_r, # reflect consistency of rollout frames. Not the error w.r.t. gt.
        'pre_x_err_avg': pre_x_err_avg,
        'post_x_err_avg': post_x_err_avg,
        'pre_vel_err_avg': pre_vel_err_avg,
        'post_vel_err_avg': post_vel_err_avg,
        'pre_momentum_error_avg': pre_momentum_error_avg,
        'post_momentum_error_avg': post_momentum_error_avg,
        'pre_energy_error_avg': pre_energy_error_avg,
        'post_energy_error_avg': post_energy_error_avg,
    }



def get_gt_and_rollout_videos(video_path, size=256):
    # Capture video from file
    cap = cv2.VideoCapture(str(video_path))

    gt_video = []
    recon_video = []
    rollout_video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split the frame into three parts
        # print(frame.shape)
        frame1 = frame[:, :size, :]
        frame2 = frame[:, size:size*2, :]
        frame3 = frame[:, size*2:, :]
        # print(frame.shape)
        # print(frame1.shape)
        # print(frame2.shape)
        # print(frame3.shape)
        
        gt_video.append(frame1)
        recon_video.append(frame2)
        rollout_video.append(frame3)

    # Resize frames
    # resized_frame1 = cv2.resize(frame1, (512, 512), interpolation=cv2.INTER_AREA)
    # resized_frame3 = cv2.resize(frame3, (512, 512), interpolation=cv2.INTER_AREA)

    # Release the video capture/writer
    cap.release()

    gt_video = np.stack(gt_video, axis=0)[:,  ::-1, :, ::-1]
    recon_video = np.stack(recon_video, axis=0)[:,  ::-1, :, ::-1]
    rollout_video = np.stack(rollout_video, axis=0)[:, ::-1, :, ::-1]

    # Return the paths to the videos
    return gt_video, recon_video, rollout_video


def get_item(f, meta_index='00000', local_index=990):
    gt_frames_stream = f['video_streams'][meta_index][local_index]
    gt_frames = iio.imread(gt_frames_stream.tobytes(), index=None, format_hint='.mp4')
    # vertical flip
    gt_frames = gt_frames[:, ::-1]
    gt_states = f['position_streams'][meta_index][local_index]
    gt_init = f['init_streams'][meta_index][local_index]
    # assume gt_frames as rollout_frames
    rollout_frames = gt_frames
    return rollout_frames, gt_states, gt_init


def eval_parabola():
    type == 'parabola'
    models = [
        # 'gt',
        # 'recon',
        # 'gt_hdf5',
        # 'dit_b_300k',
        'visdata-dit_b_300k/Step100K',
        
        # 'dit_s_30k/Step100K',
        # 'dit_s_300k/Step100K',
        # 'dit_s_3M/Step100K',
        # 'dit_b_30k/Step100K',
        # 'dit_b_3M/Step100K',
        # 'dit_l_30k/Step100K',
        # 'dit_l_300k/Step100K',
        # 'dit_l_3M/Step100K',
        
        
    ]
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/parabola/vis_data_L')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/parabola/vis_data_L.hdf5'
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/parabola/visdata')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/parabola/vis_data.hdf5'
    enable_y = True

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4


    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')

    for model in models:
        if model in ['gt', 'gt_hdf5', 'recon']:
            data_path = eval_data_dir / f'visdata-dit_b_300k/Test/'
        else:
            data_path = eval_data_dir / model
            if list(data_path.glob('*.mp4')) == 0:
                data_path = eval_data_dir / 'Test'
                
        if 'dit_l' in model or 'dit_s' in model \
            or 'dit_b_30k' in model or 'dit_b_3M' in model:
            frame_size = 128
        else:
            frame_size = 256
        
        video_list = sorted(list(data_path.glob('*.mp4')))
        assert len(video_list) > 0, data_path
        record_list = []

        for video_path in tqdm(video_list):
            # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
            gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path, size=frame_size)
            meta_index = video_path.name[:5]
            local_index = int(video_path.name[6:11])
            gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)

            if model == 'gt':
                frames = gt_frames
            elif model == 'gt_hdf5':
                frames = gt_hdf5_frames[:-1]
            elif model == 'recon':
                frames = recon_frames
            else:
                frames = rollout_frames

            ret = evaluate_xy(frames, gt_states[:-1], gt_init, mode='all')
            ret['name'] = f'{meta_index}_{local_index}'
            record_list.append(ret)


        models_metrics[model] = record_list
    f.close()
    
    for model, record_list in models_metrics.items():
        for x in record_list:
            x['x_vel_error'] = np.abs(x['x_vel'].mean()  - x['init'][1])


    # print(record_list)
    keys = [
        'delta_r',
        'abs_x_err_avg',
        # 'abs_x_vel_err_avg',
        # 'relative_x_err_avg',
        # 'relative_x_vel_err_avg',
    ]
    if enable_y:
        keys.extend([
            'abs_y_err_avg',
            # 'abs_y_vel_err_avg',
            # 'relative_y_err_avg',
            # 'relative_y_vel_err_avg',
        ])

    data = {key: [] for key in ['model']+keys}


    for model, record_list in models_metrics.items():
        merged_dict = {}
        for key in keys:
            merged_dict[key] = np.array([x[key] for x in record_list])


        # split in dist and out of dist
        # set training range
        in_indices = []
        out_indices = []
        r_out_indices = []
        v_out_indices = []
        rv_out_indices = []
        zero_indices = []

        for i, record in enumerate(record_list):

            # if record['init'][0] < 0.3: # filter out dispearing balls
            if record['init'][0] < 0.6: # filter out dispearing balls
                continue
            
            if record['init'][1] == 0:
                zero_indices.append(i)
            elif MIN_R <= record['init'][0] <= MAX_R and MIN_V <= record['init'][1] <= MAX_V:
                in_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and MIN_V <= record['init'][1] <= MAX_V:
                r_out_indices.append(i)
            elif (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
                v_out_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
                rv_out_indices.append(i)
            else:
                raise ValueError('Unexpected case')
            out_indices = zero_indices +  r_out_indices + v_out_indices + rv_out_indices


        if i == 0:  
            print('in dist, ', len(in_indices))
            print('out dist, ', len(out_indices))
            print('zero dist, ', len(zero_indices))
            print('r out dist, ', len(r_out_indices))
            print('v out dist, ', len(v_out_indices))
            print('r & v out dist, ', len(rv_out_indices))
            print('total, ', len(record_list))
            # assert len(zero_indices) + len(in_indices) + len(r_out_indices) + len(v_out_indices) \
                # + len(rv_out_indices) == len(record_list)

        print('Eval:', model)

        for key in keys:
            in_dist_mean = np.nanmean(merged_dict[key][in_indices])
            out_dist_mean = np.nanmean(merged_dict[key][out_indices])
            r_out_dist_mean = np.nanmean(merged_dict[key][r_out_indices])
            v_out_dist_mean = np.nanmean(merged_dict[key][v_out_indices])
            rv_out_dist_mean = np.nanmean(merged_dict[key][rv_out_indices])
            zero_dist_mean = np.nanmean(merged_dict[key][zero_indices])
            # data[key].append('{:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, zero_dist_mean))
            data[key].append('{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, r_out_dist_mean, v_out_dist_mean, rv_out_dist_mean, zero_dist_mean))
        data['model'].append(model)

    for key in keys:
        data[key].append('{:6s}, {:6s}, {:6s}, {:6s}, {:6s}, {:6s}'.format('in', 'out', 'r_out', 'v_out', 'rv_out', 'v_zero'))
    data['model'].append('')
    for key in keys:
        data[key].append('{:6d}, {:6d}, {:6d}, {:6d}, {:6d}, {:6d}'.format(len(in_indices), len(out_indices), len(r_out_indices), len(v_out_indices), len(rv_out_indices), len(zero_indices)))
    data['model'].append('')

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Define the CSV file path
    csv_file_path = f'{type}_models_metrics_{len(video_list)}.csv'

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")

    n = 10
    for model in models:
        record_list = models_metrics[model]
        print(f'\n{model:}')
        # r_err_array = np.array([x['delta_r'] for x in record_list])
        # print(np.argsort(r_err_array)[-n:][::-1])
        pre_vel_err_array = np.array([x['abs_x_err_avg'] for x in record_list])
        print(np.argsort(pre_vel_err_array)[-n:][::-1])


def eval_circle():
    type == 'circle'

    models = [
        'gt',
        # 'gt_hdf5',
        '1+3balls_dit_b_600k',
        'uni+static_dit_b_460k',
        'allballs_dit_b_2.1m',
        'allballs+allstatic_dit_b_2.4m',

    ]
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/visdata_circle_L')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/moving_objects/vis_data_circle_L.hdf5'
    enable_y = True

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4


    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')

    for model in models:
        if model in ['gt', 'gt_hdf5', 'recon']:
            data_path = eval_data_dir / f'visdata-{models[-1]}/Test/'
        else:
            data_path = eval_data_dir / f'visdata-{model}/Test/'
        
        video_list = sorted(list(data_path.glob('*.mp4')))
        record_list = []

        for video_path in tqdm(video_list):
            # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
            gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path)
            meta_index = video_path.name[:5]
            local_index = int(video_path.name[6:11])
            gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)

            if model == 'gt':
                frames = gt_frames
            elif model == 'gt_hdf5':
                frames = gt_hdf5_frames[:-1]
            elif model == 'recon':
                frames = recon_frames
            else:
                frames = rollout_frames

            ret = evaluate_xy(frames, gt_states[:-1], gt_init, mode='all')
            record_list.append(ret)


        models_metrics[model] = record_list
    f.close()


    # print(record_list)
    keys = [
        'delta_r',
        'abs_x_err_avg',
        # 'abs_x_vel_err_avg',
        # 'relative_x_err_avg',
        # 'relative_x_vel_err_avg',
    ]
    if enable_y:
        keys.extend([
            'abs_y_err_avg',
            # 'abs_y_vel_err_avg',
            # 'relative_y_err_avg',
            # 'relative_y_vel_err_avg',
        ])

    data = {key: [] for key in ['model']+keys}


    for model, record_list in models_metrics.items():
        merged_dict = {}
        for key in keys:
            merged_dict[key] = np.array([x[key] for x in record_list])


        # split in dist and out of dist
        # set training range
        in_indices = []
        r_out_indices = []
        v_out_indices = []
        rv_out_indices = []
        zero_indices = []

        for i, record in enumerate(record_list):

            if record['init'][0] < 0.5: # filter out dispearing balls
                continue

            if record['init'][2] == 0: #! (r, r, v)
                zero_indices.append(i)
            elif MIN_R <= record['init'][0] <= MAX_R and MIN_V <= record['init'][2] <= MAX_V:
                in_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and MIN_V <= record['init'][2] <= MAX_V:
                r_out_indices.append(i)
            elif (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][2] <= MAX_V):
                v_out_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][2] <= MAX_V):
                rv_out_indices.append(i)
            else:
                raise ValueError('Unexpected case')

        # print('zero dist, ', len(zero_indices))
        # print('in dist, ', len(in_indices))
        # print('r out dist, ', len(r_out_indices))
        # print('v out dist, ', len(v_out_indices))
        # print('r & v out dist, ', len(rv_out_indices))
        # print('total, ', len(record_list))
        # # assert len(zero_indices) + len(in_indices) + len(r_out_indices) + len(v_out_indices) \
        #     # + len(rv_out_indices) == len(record_list)

        print('Eval:', model)

        for key in keys:
            in_dist_mean = np.nanmean(merged_dict[key][in_indices])
            r_out_dist_mean = np.nanmean(merged_dict[key][r_out_indices])
            # data[key].append('{:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, zero_dist_mean))
            data[key].append('{:.4f}'.format(r_out_dist_mean))

        data['model'].append(model)

    for key in keys:
        data[key].append('{:6s}'.format('r_out'))
    data['model'].append('')

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Define the CSV file path
    csv_file_path = f'{type}_models_metrics_{len(video_list)}.csv'

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")

    n = 10
    for model in models:
        record_list = models_metrics[model]
        print(f'\n{model:}')
        # r_err_array = np.array([x['delta_r'] for x in record_list])
        # print(np.argsort(r_err_array)[-n:][::-1])
        pre_vel_err_array = np.array([x['abs_x_err_avg'] for x in record_list])
        print(np.argsort(pre_vel_err_array)[-n:][::-1])
  

def eval_uniform():
    type == 'uniform'

    models = [
        # 'gt',
        # 'recon',
        # 'dit_b_30k',
        # 'dit_b_300k',
        # '3balls_dit_b_30k',
        # '3balls_dit_b_300k',
        # '1+3balls_dit_b_600k',
        # 'allballs_dit_b_2.4m',
        # # 'resmlp_s_30kdata_100kstep',
        # 'resmlp_b_30kdata_100kstep',
        # # 'resmlp_l_30kdata_100kstep',
        # # 'resmlp_s_300kdata_100kstep',
        # 'resmlp_b_300kdata_100kstep',
        # # 'resmlp_l_300kdata_100kstep',
        
        # 'dit_s_30k/Step100K',
        # 'dit_s_300k/Step100K',
        # 'dit_s_3M/Step100K',
        'dit_b_30k/Step100K',
        # 'dit_b_300k/Step100K',
        # 'allballs_dit_b_2.4m/Step100K',
        # 'dit_l_30k/Step100K',
        # 'dit_l_300k/Step50K',
        # 'dit_l_300k/Step100K',
        # 'dit_l_3M/Step100K',
    ]
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/vis_data_L')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/vis_data_L.hdf5'
    enable_y = True

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4


    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')
    num_samples = 0

    for model in models:
        if model in ['gt', 'gt_hdf5', 'recon']:
            data_path = eval_data_dir / f'dit_b_300k/Test/'
        else:
            data_path = eval_data_dir / model
            if list(data_path.glob('*.mp4')) == 0:
                data_path = eval_data_dir / 'Test'
                
        if 'dit_l' in model or 'dit_s' in model:
            frame_size = 128
        else:
            frame_size = 256
        
        record_list = []
        if 'dt' in model or 'mlp' in model: # decision transftomer
            pred_f = h5py.File(os.path.join(data_path, 'dt_inference.hdf5'), 'r')
            preds = pred_f['pred']
            count = 0
            for meta_index in preds.keys():
                pred_ds = preds[meta_index]
                gt_ds = f['position_streams'][meta_index]
                count += len(gt_ds)

                for local_index, gt_state in enumerate(gt_ds):
                    pred_state = pred_ds[local_index]
                    gt_init = f['init_streams'][meta_index][local_index]
                    
                    # try:
                    ret = evaluate_xy(None, gt_state, gt_init, mode='all', pred_states=pred_state)
                    ret['name'] = f'{meta_index}_{local_index}'
                    record_list.append(ret)
                    # except:
                    #     print(f'{meta_index}_{local_index}')
            num_samples = count    

        else:
            video_list = sorted(list(data_path.glob('*.mp4')))
            assert len(video_list) > 0, data_path
            num_samples = len(video_list)

            for video_path in tqdm(video_list):
                # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
                gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path, size=frame_size)
                meta_index = video_path.name[:5]
                local_index = int(video_path.name[6:11])
                gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)

                if model == 'gt':
                    frames = gt_frames
                elif model == 'gt_hdf5':
                    frames = gt_hdf5_frames[:-1]
                elif model == 'recon':
                    frames = recon_frames
                else:
                    frames = rollout_frames

                ret = evaluate_xy(frames, gt_states[:-1], gt_init, mode='all')
                ret['name'] = f'{meta_index}_{local_index}'
                record_list.append(ret)


        models_metrics[model] = record_list
    f.close()
    
    # save record
    # import pickle
    # pkl_file_path = f'uniform_motion_vis_data_L.pkl'
    # # Check if the pickle file exists
    # if os.path.exists(pkl_file_path):
    #     # If the file exists, read the existing dictionary
    #     with open(pkl_file_path, 'rb') as file:
    #         existing_dict = pickle.load(file)
        
    #     # Add new keys to the existing dictionary
    #     existing_dict.update(models_metrics)
    # else:
    #     # If the file does not exist, use the new dictionary as the initial dictionary
    #     existing_dict = models_metrics

    # # Save the updated dictionary back to the pickle file
    # with open(pkl_file_path, 'wb') as file:
    #     pickle.dump(existing_dict, file)
    # exit()

    for model, record_list in models_metrics.items():
        for x in record_list:
            x['vel_error'] = np.abs(x['x_vel'].mean()  - x['init'][1])
            
            # print(x['x_vel'])
            # print(x['x_vel'].mean())
            # print(x['init'])
            # print(x['vel_error'])
            # exit()

    # print(record_list)
    keys = [
        'vel_error',
        'delta_r',
        'abs_x_err_avg',
        # 'abs_x_vel_err_avg',
        # 'relative_x_err_avg',
        # 'relative_x_vel_err_avg',
    ]
    if enable_y:
        keys.extend([
            'abs_y_err_avg',
            # 'abs_y_vel_err_avg',
            # 'relative_y_err_avg',
            # 'relative_y_vel_err_avg',
        ])


    data = {key: [] for key in ['model']+keys}


    for model, record_list in models_metrics.items():
        merged_dict = {}
        for key in keys:
            merged_dict[key] = np.array([x[key] for x in record_list])


        # split in dist and out of dist
        # set training range
        in_indices = []
        out_indices = []
        r_out_indices = []
        v_out_indices = []
        rv_out_indices = []
        zero_indices = []

        for i, record in enumerate(record_list):

            if record['init'][0] < 0.6: # filter out dispearing balls
                continue
            
            if record['init'][1] == 0: #! (r, v)
                zero_indices.append(i)
            elif MIN_R <= record['init'][0] <= MAX_R and MIN_V <= record['init'][1] <= MAX_V:
                in_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and MIN_V <= record['init'][1] <= MAX_V:
                r_out_indices.append(i)
            elif (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
                v_out_indices.append(i)
            elif not (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
                rv_out_indices.append(i)
            else:
                raise ValueError('Unexpected case')
            out_indices = zero_indices +  r_out_indices + v_out_indices + rv_out_indices

        all_indices = in_indices + r_out_indices + v_out_indices + rv_out_indices + zero_indices   
        if model == models[0]:
            print('in dist, ', len(in_indices))
            print('out dist, ', len(out_indices))
            print('zero dist, ', len(zero_indices))
            print('r out dist, ', len(r_out_indices))
            print('v out dist, ', len(v_out_indices))
            print('r & v out dist, ', len(rv_out_indices))
            print('')

        print('Eval:', model)

        for key in keys:
            in_dist_mean = np.nanmean(merged_dict[key][in_indices])
            out_dist_mean = np.nanmean(merged_dict[key][out_indices])
            r_out_dist_mean = np.nanmean(merged_dict[key][r_out_indices])
            v_out_dist_mean = np.nanmean(merged_dict[key][v_out_indices])
            rv_out_dist_mean = np.nanmean(merged_dict[key][rv_out_indices])
            zero_dist_mean = np.nanmean(merged_dict[key][zero_indices])
            # data[key].append('{:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, zero_dist_mean))
            data[key].append('{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, r_out_dist_mean, v_out_dist_mean, rv_out_dist_mean, zero_dist_mean))

        data['model'].append(model)


    for key in keys:
        data[key].append('{:6s}, {:6s}, {:6s}, {:6s}, {:6s}, {:6s}'.format('in', 'out', 'r_out', 'v_out', 'rv_out', 'v_zero'))
    data['model'].append('')
    for key in keys:
        data[key].append('{:6d}, {:6d}, {:6d}, {:6d}, {:6d}, {:6d}'.format(len(in_indices), len(out_indices), len(r_out_indices), len(v_out_indices), len(rv_out_indices), len(zero_indices)))
    data['model'].append('')

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Define the CSV file path
    csv_file_path = f'{type}_models_vel_metrics_{num_samples}.csv'

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")

    n = 10
    for model in models:
        record_list = [models_metrics[model][i] for i in all_indices] # remove 
        print(f'\n{model:}')
        # r_err_array = np.array([x['delta_r'] for x in record_list])
        # print(np.argsort(r_err_array)[-n:][::-1])
        x_err_array = np.array([x['abs_x_err_avg'] for x in record_list])
        index = np.argsort(x_err_array)[-n:][::-1].tolist()
        print(np.array([x['name'] for x in record_list])[index].tolist()) 


def eval_collision():

    models = [
    # 'gt',
    'recon',
    # 'dit_s_10k',
    # 'dit_s_110k',
    # 'dit_s_1m',
    # 'dit_b_10k',
    # 'dit_b_110k',
    # 'dit_b_1m',
    # 'dit_l_10k',
    # 'dit_l_110k',
    # 'dit_l_1m',
    # 'dit_xl_1m',
    # "mmdit_b_1m",
    # "mmdit_text_b_1m"
    # 'dt_s_lr1e-4_10k_steps_1.6m',
    # 'dt_s_lr1e-4_20k_steps_1.6m',
    # 'dt_s_lr1e-4_30k_steps_1.6m',
    # 'dt_s_lr1e-4_40k_steps_1.6m',
    # 'dt_s_lr1e-4_50k_steps_1.6m',
    # 'dt_b_lr1e-4_10k_steps_1.6m',
    # 'dt_b_lr1e-4_20k_steps_1.6m',
    # 'dt_b_lr1e-4_30k_steps_1.6m',
    # 'dt_b_lr1e-4_40k_steps_1.6m',
    # 'dt_b_lr1e-4_50k_steps_1.6m',
    # 'dt_l_lr1e-4_10k_steps_1.6m',
    # 'dt_l_lr1e-4_20k_steps_1.6m',
    # 'dt_l_lr1e-4_30k_steps_1.6m',
    # 'dt_l_lr1e-4_40k_steps_1.6m',
    # 'dt_l_lr1e-4_50k_steps_1.6m',
    # 'encoder_dt_b_lr1e-4_8k_steps_1.6m',
    # 'encoder_dt_b_lr1e-4_10k_steps_1.6m',
    # 'encoder_dt_b_lr1e-4_26k_steps_1.6m',
    # 'encoder_dt_b_lr1e-4_36k_steps_1.6m',
    # 'encoder_dt_b_lr1e-4_50k_steps_1.6m',
    # 'encoder_dt_b_lr1e-5_10k_steps_1.6m',
    # 'encoder_dt_b_lr1e-5_20k_steps_1.6m',
    # 'encoder_dt_b_lr1e-5_30k_steps_1.6m',
    # 'encoder_dt_b_lr1e-5_40k_steps_1.6m',
    # 'encoder_dt_b_lr1e-5_50k_steps_1.6m',
    # 'encoder_dt_l_lr1e-5_20k_steps',
    # 'encoder_dt_l_lr1e-5_30k_steps',
    # 'encoder_dt_l_lr1e-5_50k_steps',
    # 'resmlp_b_lr1e-4_100k_steps',
    # 'resmlp_b_lr3e-4_100k_steps',
    # 'resmlp_b_lr3e-4_100k_steps_pred_all',
    # 'resmlp_b_lr1e-5_100k_steps',
    # 'resmlp_b_lr3e-5_100k_steps',
    # 'resmlp_l_lr1e-4_100k_steps',
    # 'resmlp_l_lr3e-4_100k_steps',
    # 'resmlp_l_lr3e-5_100k_steps',
    ]
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/collision/vis_data_L')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/vis_data_L.hdf5'
    enable_y = False

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.5
    MAX_R = 1.5

    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')
    num_samples = 0

    for model in models:
        if model in ['gt', 'gt_hdf5', 'recon']:
            data_path = eval_data_dir / f'dit_b_1m/Test/'
        else:
            data_path = eval_data_dir / f'{model}/Test/'

        print(model)
        record_list = []
        if 'dt' in model or 'mlp' in model: # decision transftomer
            pred_f = h5py.File(os.path.join(data_path, 'dt_inference.hdf5'), 'r')
            preds = pred_f['pred']
            count = 0
            for meta_index in preds.keys():
                pred_ds = preds[meta_index]
                gt_ds = f['position_streams'][meta_index]
                count += len(gt_ds)

                for local_index, gt_state in enumerate(gt_ds):
                    pred_state = pred_ds[local_index]
                    gt_init = f['init_streams'][meta_index][local_index]
                    
                    try:
                        ret = evaluate_xy_collision(None, gt_state, gt_init, mode='all', pred_states=pred_state)
                        ret['name'] = f'{meta_index}_{local_index}'
                        record_list.append(ret)
                    except:
                        print(f'{meta_index}_{local_index}')
            num_samples = count    
        else:
            video_list = sorted(list(data_path.glob('*.mp4')))
            assert len(video_list) > 0, data_path
            num_samples = len(video_list)

            # print(len(video_list))
            # continue
            for video_path in tqdm(video_list):
                # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
                gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path)
                meta_index = video_path.name[:5]
                local_index = int(video_path.name[6:11])
                gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)

                if model == 'gt':
                    frames = gt_frames
                elif model == 'gt_hdf5':
                    frames = gt_hdf5_frames[:-1]
                    # print(frames.shape)
                elif model == 'recon':
                    frames = recon_frames
                else:
                    frames = rollout_frames
                # print(frames.shape, gt_states.shape)
                # print(video_path.name)

                # print(gt_states)
                try:
                    ret = evaluate_xy_collision(frames, gt_states[:-1], gt_init, mode='all')
                    ret['name'] = f'{meta_index}_{local_index}'

                    record_list.append(ret)
                except:
                    print(video_path)

                # if delta_r > 0.1:
                #     print(video_path, ' failed to keep shape')


        models_metrics[model] = record_list
    f.close()

    # save record
    # import pickle
    # pkl_file_path = f'collision_models_vel_metrics_vis_data_L.pkl'
    # # Check if the pickle file exists
    # if os.path.exists(pkl_file_path):
    #     # If the file exists, read the existing dictionary
    #     with open(pkl_file_path, 'rb') as file:
    #         existing_dict = pickle.load(file)
        
    #     # Add new keys to the existing dictionary
    #     existing_dict.update(models_metrics)
    # else:
    #     # If the file does not exist, use the new dictionary as the initial dictionary
    #     existing_dict = models_metrics

    # # Save the updated dictionary back to the pickle file
    # with open(pkl_file_path, 'wb') as file:
    #     pickle.dump(existing_dict, file)

    # print(record_list)
    
    for model, record_list in models_metrics.items():
        for x in record_list:
            x['pre_vel_error'] = np.abs(x['pre_x_vel'].mean(-2)  - x['gt_pre_x_vel'].mean(-2)).mean()
            x['post_vel_error'] = np.abs(x['post_x_vel'].mean(-2)  - x['gt_post_x_vel'].mean(-2)).mean()
            
            # print(x['pre_x_vel'].mean(-2))
            # print(x['gt_pre_x_vel'].mean(-2))
            # print(x['pre_vel_error'])
            # print(x['post_x_vel'].mean(-2))
            # print(x['gt_post_x_vel'].mean(-2))
            # print(x['post_vel_error'])
            # exit()
    
    keys = [
        'delta_r',
        'pre_vel_error',
        'post_vel_error',
        'pre_x_err_avg',
        'post_x_err_avg',
        'pre_vel_err_avg',
        'post_vel_err_avg',
        'pre_momentum_error_avg',
        'post_momentum_error_avg',
        'pre_energy_error_avg',
        'post_energy_error_avg',
    ]
    if enable_y:
        keys.extend([
            'abs_y_err_avg',
            # 'abs_y_vel_err_avg',
            # 'relative_y_err_avg',
            # 'relative_y_vel_err_avg',
        ])

    in_indices = []
    out_indices = []
    r_out_indices = []
    v_out_indices = []
    rv_out_indices = []
    zero_indices = []

    for i, record in enumerate(record_list):

        if record['init'][0] <= 0.5 or record['init'][1] <=0.5: # filter out dispearing balls
            continue
        
        # r1, r2, v1, v2
        if record['init'][2] == 0 or record['init'][3] == 0:
            zero_indices.append(i)
        elif (MIN_R <= record['init'][0] <= MAX_R and MIN_R <= record['init'][1] <= MAX_R) and \
           (MIN_V <= record['init'][2] <= MAX_V and MIN_V <= record['init'][3] <= MAX_V):
            in_indices.append(i)
        elif not (MIN_R <= record['init'][0] <= MAX_R and MIN_R <= record['init'][1] <= MAX_R) and \
           (MIN_V <= record['init'][2] <= MAX_V and MIN_V <= record['init'][3] <= MAX_V):
            r_out_indices.append(i)
        elif (MIN_R <= record['init'][0] <= MAX_R and MIN_R <= record['init'][1] <= MAX_R) and \
           not (MIN_V <= record['init'][2] <= MAX_V and MIN_V <= record['init'][3] <= MAX_V):
            v_out_indices.append(i)
        elif not (MIN_R <= record['init'][0] <= MAX_R and MIN_R <= record['init'][1] <= MAX_R) and \
           not (MIN_V <= record['init'][2] <= MAX_V and MIN_V <= record['init'][3] <= MAX_V):
            rv_out_indices.append(i)
        else:
            raise ValueError('Unexpected case')
        out_indices = zero_indices +  r_out_indices + v_out_indices + rv_out_indices

    print('in dist, ', len(in_indices))
    print('out dist, ', len(out_indices))
    print('zero dist, ', len(zero_indices))
    print('r out dist, ', len(r_out_indices))
    print('v out dist, ', len(v_out_indices))
    print('r & v out dist, ', len(rv_out_indices))
    print('')

    data = {key: [] for key in ['model']+keys}

    for model, record_list in models_metrics.items():
        merged_dict = {}
        for key in keys:
            merged_dict[key] = np.array([x[key] for x in record_list])


        print('Eval:', model)

        for key in keys:
            in_dist_mean = np.nanmean(merged_dict[key][in_indices])
            out_dist_mean = np.nanmean(merged_dict[key][out_indices])
            r_out_dist_mean = np.nanmean(merged_dict[key][r_out_indices])
            v_out_dist_mean = np.nanmean(merged_dict[key][v_out_indices])
            rv_out_dist_mean = np.nanmean(merged_dict[key][rv_out_indices])
            zero_dist_mean = np.nanmean(merged_dict[key][zero_indices])
            # data[key].append('{:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, zero_dist_mean))
            data[key].append('{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, r_out_dist_mean, v_out_dist_mean, rv_out_dist_mean, zero_dist_mean))

        data['model'].append(model)

    for key in keys:
        data[key].append('{:6s}, {:6s}, {:6s}, {:6s}, {:6s}, {:6s}'.format('in', 'out', 'r_out', 'v_out', 'rv_out', 'v_zero'))
    data['model'].append('')
    for key in keys:
        data[key].append('{:6d}, {:6d}, {:6d}, {:6d}, {:6d}, {:6d}'.format(len(in_indices), len(out_indices), len(r_out_indices), len(v_out_indices), len(rv_out_indices), len(zero_indices)))
    data['model'].append('')

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Define the CSV file path
    csv_file_path = f'collision_models_vel_metrics_{num_samples}.csv'

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")

    n = 10
    for model in models:
        record_list = models_metrics[model]
        print(f'\n{model:}')
        # r_err_array = np.array([x['delta_r'] for x in record_list])
        # print(np.argsort(r_err_array)[-n:][::-1])
        post_x_err_array = np.array([x['post_x_err_avg'] for x in record_list])
        index = np.argsort(post_x_err_array)[-n:][::-1].tolist()
        print(np.array([x['name'] for x in record_list])[index].tolist()) 
        

def eval_collision_square_out():


    frame_size = 128

    # vel
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/collision/square_out_eval_vel_6.8K')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_vel_6.8K.hdf5'
    pkl_file_path = f'collision_square_out_eval_vel_6.8K.pkl'
    # model, frame_size = 'dit_b_1m/Step300K', 256 # no square
    # model = 'Step50K' # vel 2-3
    # model = 'Step60K' # vel 2-3
    # model = 'square_out_1.4M/Step100K' # vel 2-3
    # model='square_out_vel_1.25-3.75_493.9K/Step50K' # vel 1.25-3.75
    # model='square_out_vel_1.25-3.75_493.9K/Step90K' # vel 1.25-3.75
    # model='square_out_vel_1.1-3.9_258.0K/Step50K' # vel 1.25-3.75
    # model='square_outhalf_vel_1.25_306.9K/Step50K' # vel 1.25-3.75
    model='square_outhalf_vel_1.25_306.9K/Step100K' # vel 1.25-3.75
    
    # time composition
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/time_composition/collision_bounce_eval_54_v2')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/time_composition/collision_bounce_eval_54_v2.hdf5'
    # pkl_file_path = f'collision_bounce_eval_54_v2.pkl'
    # model = 'collision_bounce_237.3K_v3/Step50K'
    
    data_path = eval_data_dir / model
    
    
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/collision/square_out_eval_vel_6.8K/square_out_1.4M')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_vel_6.8K.hdf5'
    # pkl_file_path = f'collision_square_out_eval_vel_6.8K.pkl'
    # model = 'Step100K'
    # data_path = eval_data_dir / model
    
    # multiple vel
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/collision/square_out_eval_vel_6.8K-vel_multiple_1.2M')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_vel_6.8K.hdf5'
    # pkl_file_path = f'collision_square_out_eval_vel_multiple_6.8K.pkl'
    # # model = 'Step80K'
    # model = 'Step100K'
    # data_path = eval_data_dir / model
    
    # mass
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/collision/square_out_eval_mass_5.9K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/collision/square_out_eval_mass_5.9K.hdf5'
    # pkl_file_path = f'collision_square_out_eval_mass_5.9K.pkl'
    # # model = 'Step70K'
    # model = 'Step100K'
    # data_path = eval_data_dir / model
    
    
    enable_y = False

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.5
    MAX_R = 1.5

    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')
    num_samples = 0



    record_list = []
    video_list = sorted(list(data_path.glob('*.mp4')))
    assert len(video_list) > 0, data_path
    num_samples = len(video_list)

    # print(len(video_list))
    # continue
    for video_path in tqdm(video_list):
        # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
        gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path, size=frame_size)
        meta_index = video_path.name[:5]
        local_index = int(video_path.name[6:11])
        gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)
        if model == 'gt':
            frames = gt_frames
        elif model == 'gt_hdf5':
            frames = gt_hdf5_frames[:-1]
        elif model == 'recon':
            frames = recon_frames
        else:
            frames = rollout_frames

        try:
            ret = evaluate_xy_collision(frames, gt_states[:-1], gt_init, mode='all')
            ret['name'] = f'{meta_index}_{local_index}'
            if ret['init'][0] <= 0.5 or ret['init'][1] <=0.5:
                continue

            record_list.append(ret)
        except:
            print(video_path)

    models_metrics[model] = record_list
    f.close()

    # save record
    import pickle
    # pkl_file_path = f'collision_square_out_eval_vel_683.pkl'
    # Check if the pickle file exists
    if os.path.exists(pkl_file_path):
        # If the file exists, read the existing dictionary
        with open(pkl_file_path, 'rb') as file:
            existing_dict = pickle.load(file)
        
        # Add new keys to the existing dictionary
        existing_dict.update(models_metrics)
    else:
        # If the file does not exist, use the new dictionary as the initial dictionary
        existing_dict = models_metrics

    # Save the updated dictionary back to the pickle file
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(existing_dict, file)



def eval_uniform_square_out():
    type == 'uniform'
    frame_size = 128

    # square in and out
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/square_out_eval_4.4K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/square_out_eval_4.4K.hdf5'
    # pkl_file_path = f'uniform_motion_square_out_eval_4.4K.pkl'
    
    # model, frame_size = 'dit_b_300k/Step300K', 256
    # model = 'Step50K'
    # model = 'square_out_1.25-3.75_200.7K/Step50K'
    # model = 'square_out_1.5-3.5_100.2K/Step50K'
    # model = 'square_out_1.75-3.25_150.0K/Step50K'
    # model = 'square_out_n=3_168.3K/Step50K'
    # model = 'square_out_n=4_100.2K/Step60K'
    # data_path = eval_data_dir / model
    
    # extraploate test
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/square_out_eval_extrapolate_5.8K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/square_out_eval_extrapolate_5.8K.hdf5'
    # pkl_file_path = f'uniform_motion_square_out_eval_extrapolate_5.8K.pkl'
    
    # model, frame_size = 'dit_b_300k/Step50K', 256
    # model, frame_size = 'dit_b_300k/Step300K', 256
    # model = 'Step50K'
    # model = 'square_out_1.25-3.75_200.7K/Step50K'
    # model = 'square_out_1.5-3.5_100.2K/Step50K'
    # model = 'square_out_1.75-3.25_150.0K/Step50K'
    # model = 'square_out_n=3_168.3K/Step50K'
    # model = 'square_out_n=4_100.2K/Step60K'
    # data_path = eval_data_dir / model
    
    # size/vel/shape/color
    
    # size-vel
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/square_out_eval_4.4K/size_vel_150K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/square_out_eval_4.4K.hdf5'
    # pkl_file_path = f'uniform_motion_size_vel.pkl'
    # model = 'Step90K'
    
    # color-size
    eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/color_size_eval_1.4K/')
    hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/color_size_eval_1.4K.hdf5'
    pkl_file_path = f'uniform_motion_color_size.pkl'
    model = 'color_size_300K_cogvae/Step60K'
    
    # # # color-vel
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/color_vel_eval_1.6K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/color_vel_eval_1.6K.hdf5'
    # pkl_file_path = f'uniform_motion_color_vel.pkl'
    # model = 'color_vel_300K/Step50K'
    # model = 'color_vel_300K_noflip/Step50K'
    # model = 'baseline1_300.0K/Step60K'
    # model = 'baseline2_300.0K/Step50K'
    
    # color-shape
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/color_shape_eval_1.4K/color_shape_600K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/color_shape_eval_1.4K.hdf5'
    # pkl_file_path = f'uniform_motion_color_shape.pkl'
    # model = 'Step50K'
    
    # vel-shape
    # eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/uniform_motion/vel_shape_eval_704/vel_shape_300K')
    # hdf5_path = '/mnt/bn/bykang/phy-data/simple_scenes_data/uniform_motion/vel_shape_eval_704.hdf5'
    # pkl_file_path = f'uniform_motion_vel_shape.pkl'
    # model = 'Step50K'
    
    data_path = eval_data_dir / model
    
    
    enable_y = True

    MIN_V = 1.0                     
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4


    models_metrics = {}
    f = h5py.File(hdf5_path, 'r')
    num_samples = 0

    # model = 'recon'
    # model = 'gt'

    
    record_list = []
    video_list = sorted(list(data_path.glob('*.mp4')))
    assert len(video_list) > 0, data_path
    num_samples = len(video_list)

    for video_path in tqdm(video_list):
        # video_path = Path('/mnt/bn/magic/ckpt/angry_world/collision/samples-dit_b_10k/Test/00004-00598.mp4')
        gt_frames,  recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path, size=frame_size)
        meta_index = video_path.name[:5]
        local_index = int(video_path.name[6:11])
        gt_hdf5_frames, gt_states, gt_init = get_item(f, meta_index, local_index)

        if model == 'gt':
            frames = gt_frames
        elif model == 'gt_hdf5':
            frames = gt_hdf5_frames[:-1]
        elif model == 'recon':
            frames = recon_frames
        else:
            frames = rollout_frames

        # ret = evaluate_xy(frames, gt_states[:-1], gt_init, mode='all')
        ret = evaluate_xy(frames, gt_states, gt_init, mode='all')
        ret['name'] = f'{meta_index}_{local_index}'
        record_list.append(ret)


    models_metrics[model] = record_list
    f.close()
    
    # save record
    import pickle
    # Check if the pickle file exists
    if os.path.exists(pkl_file_path):
        # If the file exists, read the existing dictionary
        with open(pkl_file_path, 'rb') as file:
            existing_dict = pickle.load(file)
        
        # Add new keys to the existing dictionary
        existing_dict.update(models_metrics)
    else:
        # If the file does not exist, use the new dictionary as the initial dictionary
        existing_dict = models_metrics

    # Save the updated dictionary back to the pickle file
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(existing_dict, file)


    # print(record_list)
    # keys = [
    #     'delta_r',
    #     'abs_x_err_avg',
    #     # 'abs_x_vel_err_avg',
    #     # 'relative_x_err_avg',
    #     # 'relative_x_vel_err_avg',
    # ]
    # if enable_y:
    #     keys.extend([
    #         'abs_y_err_avg',
    #         # 'abs_y_vel_err_avg',
    #         # 'relative_y_err_avg',
    #         # 'relative_y_vel_err_avg',
    #     ])


    # data = {key: [] for key in ['model']+keys}


    # for model, record_list in models_metrics.items():
    #     merged_dict = {}
    #     for key in keys:
    #         merged_dict[key] = np.array([x[key] for x in record_list])


    #     # split in dist and out of dist
    #     # set training range
    #     in_indices = []
    #     out_indices = []
    #     r_out_indices = []
    #     v_out_indices = []
    #     rv_out_indices = []
    #     zero_indices = []

    #     for i, record in enumerate(record_list):

    #         if record['init'][0] < 0.6: # filter out dispearing balls
    #             continue
            
    #         if record['init'][1] == 0: #! (r, v)
    #             zero_indices.append(i)
    #         elif MIN_R <= record['init'][0] <= MAX_R and MIN_V <= record['init'][1] <= MAX_V:
    #             in_indices.append(i)
    #         elif not (MIN_R <= record['init'][0] <= MAX_R) and MIN_V <= record['init'][1] <= MAX_V:
    #             r_out_indices.append(i)
    #         elif (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
    #             v_out_indices.append(i)
    #         elif not (MIN_R <= record['init'][0] <= MAX_R) and not (MIN_V <= record['init'][1] <= MAX_V):
    #             rv_out_indices.append(i)
    #         else:
    #             raise ValueError('Unexpected case')
    #         out_indices = zero_indices +  r_out_indices + v_out_indices + rv_out_indices

    #     all_indices = in_indices + r_out_indices + v_out_indices + rv_out_indices + zero_indices   
    #     if model == models[0]:
    #         print('in dist, ', len(in_indices))
    #         print('out dist, ', len(out_indices))
    #         print('zero dist, ', len(zero_indices))
    #         print('r out dist, ', len(r_out_indices))
    #         print('v out dist, ', len(v_out_indices))
    #         print('r & v out dist, ', len(rv_out_indices))
    #         print('')

    #     print('Eval:', model)

    #     for key in keys:
    #         in_dist_mean = np.nanmean(merged_dict[key][in_indices])
    #         out_dist_mean = np.nanmean(merged_dict[key][out_indices])
    #         r_out_dist_mean = np.nanmean(merged_dict[key][r_out_indices])
    #         v_out_dist_mean = np.nanmean(merged_dict[key][v_out_indices])
    #         rv_out_dist_mean = np.nanmean(merged_dict[key][rv_out_indices])
    #         zero_dist_mean = np.nanmean(merged_dict[key][zero_indices])
    #         # data[key].append('{:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, zero_dist_mean))
    #         data[key].append('{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(in_dist_mean, out_dist_mean, r_out_dist_mean, v_out_dist_mean, rv_out_dist_mean, zero_dist_mean))

    #     data['model'].append(model)


    # for key in keys:
    #     data[key].append('{:6s}, {:6s}, {:6s}, {:6s}, {:6s}, {:6s}'.format('in', 'out', 'r_out', 'v_out', 'rv_out', 'v_zero'))
    # data['model'].append('')
    # for key in keys:
    #     data[key].append('{:6d}, {:6d}, {:6d}, {:6d}, {:6d}, {:6d}'.format(len(in_indices), len(out_indices), len(r_out_indices), len(v_out_indices), len(rv_out_indices), len(zero_indices)))
    # data['model'].append('')

    # # Convert dictionary to DataFrame
    # df = pd.DataFrame(data)

    # # Define the CSV file path
    # csv_file_path = f'{type}_models_metrics_{num_samples}_state.csv'

    # # Save DataFrame to CSV
    # df.to_csv(csv_file_path, index=False)

    # print(f"Data saved to {csv_file_path}")

    # n = 10
    # for model in models:
    #     record_list = [models_metrics[model][i] for i in all_indices] # remove 
    #     print(f'\n{model:}')
    #     # r_err_array = np.array([x['delta_r'] for x in record_list])
    #     # print(np.argsort(r_err_array)[-n:][::-1])
    #     x_err_array = np.array([x['abs_x_err_avg'] for x in record_list])
    #     index = np.argsort(x_err_array)[-n:][::-1].tolist()
    #     print(np.array([x['name'] for x in record_list])[index].tolist()) 
  


if __name__ == '__main__':
    from sys import argv
    type = argv[1]
    if type == 'parabola':
        eval_parabola()
    elif type == 'circle':
        eval_circle()
    elif type == 'collision':
        eval_collision()
    elif type == 'uniform':
        eval_uniform()
    elif type == 'collision_square_out':
        eval_collision_square_out()
    elif type == 'uniform_square_out':
        eval_uniform_square_out()

    # models = [
    # # 'gt_hdf5',
    # 'gt',
    # # 'recon', 
    # #  'dit_s_10k',
    # #  'dit_s_110k', 'dit_s_1m', 'dit_b_10k', 'dit_b_110k', 
    # 'collision_dit_b_1m', 
    # '1ball_dit_b_300k', 
    # '3balls_dit_b_30k', 
    # '3balls_dit_b_300k', 
    # # 'dit_l_10k', 'dit_l_110k', 'dit_l_1m', 
    # #  'dit_xl_1m',
    # # 'mmdit_b_1m',
    # # 'mmdit_text_b_1m',
    # ]
    # hdf5_path = '/mnt/bn/magic/simple_scenes_data/uniform_motion/vis_data_left.hdf5'
