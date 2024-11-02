import torch
from video_metrics.calculate_fvd import calculate_fvd
from video_metrics.calculate_psnr import calculate_psnr
from video_metrics.calculate_ssim import calculate_ssim
from video_metrics.calculate_lpips import calculate_lpips
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
import os


def get_gt_and_rollout_videos(video_path):
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
        frame1 = frame[:, :256, :]
        frame2 = frame[:, 256:512, :]
        frame3 = frame[:, 512:, :]
        gt_video.append(frame1)
        recon_video.append(frame2)
        rollout_video.append(frame3)

    # Resize frames
    # resized_frame1 = cv2.resize(frame1, (512, 512), interpolation=cv2.INTER_AREA)
    # resized_frame3 = cv2.resize(frame3, (512, 512), interpolation=cv2.INTER_AREA)

    # Release the video capture/writer
    cap.release()

    gt_video = np.stack(gt_video, axis=0)[:,  :, :, ::-1]
    recon_video = np.stack(recon_video, axis=0)[:,  :, :, ::-1]
    rollout_video = np.stack(rollout_video, axis=0)[:, :, :, ::-1]

    # Return the paths to the videos
    return gt_video, recon_video, rollout_video


# ps: pixel value should be in [0, 1]!

# NUMBER_OF_VIDEOS = 8
# VIDEO_LENGTH = 30
# CHANNEL = 3
# SIZE = 64
# videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

device = torch.device("cuda")
eval_data_dir = Path('/mnt/bn/bykang/ckpts/phyworld/phyre_combo/visdata_eval_1k')
models = [
    'combo_4in8_train6_dit_xl/step680k_train',
    'combo_4in8_train6_dit_xl/step680k_train6',
    'combo_4in8_train6_dit_xl/step680k_eval',
    'combo_4in8_train60_dit_b/step1000k_train',
    'combo_4in8_train60_dit_b/step1000k_eval',
    'combo_4in8_train60_dit_xl/step800k_train',
    'combo_4in8_train60_dit_xl/step800k_eval',

]

for model in models:
    # Since FVD is sensitive to the number of videos, keep it fixed for different models on the same dataset.
    if model.endswith('train') or model.endswith('train6'):
        num_videos = 600
    elif model.endswith('eval'):
        num_videos = 500
    else:
        raise NotImplementedError
    data_path = eval_data_dir / model
    
    video_list = sorted(list(data_path.glob('*.mp4')))
    assert len(video_list) >= num_videos, data_path
    inv = int(len(video_list)/num_videos)
    video_list = video_list[::inv][:num_videos] 

    eval_num = len(video_list)


    if '/' in model:
        model_name = '_'.join(model.split('/')) + f'_eval{eval_num}'
    else:
        model_name = model + f'_eval{eval_num}'
    save_path = f'./results/{model_name}.json'
    if os.path.exists(save_path):
        continue
    
    with open(save_path, 'w') as file:

        gt, rollout = [], []
        for video_path in tqdm(video_list):
            gt_frames, recon_frames, rollout_frames = get_gt_and_rollout_videos(video_path)
            gt_frames, recon_frames, rollout_frames = gt_frames/255.0,  recon_frames/255.0, rollout_frames/255.0
            gt.append(torch.tensor(gt_frames, dtype=torch.float32).permute(0, 3, 1, 2)) # THWC -> TCHW
            rollout.append(torch.tensor(rollout_frames, dtype=torch.float32).permute(0, 3, 1, 2))
        gt = torch.stack(gt, dim=0)
        rollout = torch.stack(rollout, dim=0)

        videos1, videos2 = gt, rollout
        print(videos1.shape, videos2.shape)
        
        result = {}
        result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
        # result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
        result['ssim'] = calculate_ssim(videos1, videos2)
        result['psnr'] = calculate_psnr(videos1, videos2)
        result['lpips'] = calculate_lpips(videos1, videos2, device)
        json_str = json.dumps(result, indent=4)
        file.write(json_str)
    
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_video.py 