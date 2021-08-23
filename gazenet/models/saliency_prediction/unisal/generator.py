
import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2


def load_video_frames(frames_list, last_frame_idx, img_mean, img_std, img_width, img_height, frames_len=16):
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
        img = F.to_tensor(img)
        img = F.normalize(img, img_mean, img_std)
        frames.append(img)
    frames = torch.stack(frames, dim=0)
    return frames # .permute(1, 0, 2, 3)


def smooth_sequence(seq, method):
    shape = seq.shape

    seq = seq.reshape(shape[1], np.prod(shape[-2:]))
    if method[:3] == 'med':
        kernel_size = int(method[3:])
        ks2 = kernel_size // 2
        smoothed = np.zeros_like(seq)
        for idx in range(seq.shape[0]):
            smoothed[idx, :] = np.median(seq[max(0, idx - ks2):min(seq.shape[0], idx + ks2 + 1), :], axis=0)
        seq = smoothed.reshape(shape)
    else:
        raise NotImplementedError

    return seq