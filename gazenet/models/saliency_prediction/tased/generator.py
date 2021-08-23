import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2


def load_video_frames(frames_list, last_frame_idx, img_width, img_height, frames_len=16):
    # load video video_frames_list, process them and return a suitable tensor
    frame_number = last_frame_idx
    start_frame_number = frame_number - frames_len + 1
    start_frame_number = max(0, start_frame_number)
    frames_list_idx = [f for f in range(start_frame_number, frame_number)]
    if len(frames_list_idx) < frames_len:
        nsh = frames_len - len(frames_list_idx)
        frames_list_idx = np.concatenate((np.tile(frames_list_idx[0], (nsh)), frames_list_idx), axis=0)
    frames = []
    for i in range(len(frames_list_idx)):
        # TODO (fabawi): loading the first frame on failure is not ideal. Find a better way
        try:
            img = cv2.resize(frames_list[frames_list_idx[i]].copy(), (img_width, img_height))
        except:
            try:
                img = cv2.resize(frames_list[frames_list_idx[0]].copy(), (img_width, img_height))
            except:
                img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().mul_(2.).sub_(255).div(255)
        # img = F.to_tensor(img)
        frames.append(img)
    frames = torch.stack(frames, dim=0)
    # frames = frames.mul_(2.).sub_(255).div(255)
    return frames.permute(1, 0, 2, 3)

