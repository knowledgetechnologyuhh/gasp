import torch
import torchvision.transforms.functional as F
import numpy as np
import librosa as sf
from PIL import Image
import cv2

def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) *
                        (255.0 / (data_max - data_min)),
                0, 255).astype(np.uint8)
    return data_norm

def create_data_packet(in_data, frame_number, frames_len=16):
    in_data = np.array(in_data)
    n_frame = in_data.shape[0]
    # if the frame number is larger, we just use the last sound one heard
    frame_number = min(frame_number, n_frame)
    starting_frame = frame_number - frames_len + 1
    # ensure we do not have any negative video_frames_list
    starting_frame = max(0, starting_frame)
    data_pack = in_data
    # data_pack = in_data[starting_frame:frame_number+1, :]
    return data_pack, frames_len#frame_number


def get_wav_features(features, frame_number, frames_len=16):

    audio_data, valid_frame_number = create_data_packet(features, frame_number, frames_len=frames_len)
    return torch.from_numpy(audio_data).float().view(1,1,-1), valid_frame_number


def load_video_frames(frames_list, last_frame_idx, valid_frame_idx, img_mean, img_std, img_width, img_height, frames_len=16):
    # load video video_frames_list, process them and return a suitable tensor
    frame_number = min(last_frame_idx, valid_frame_idx)
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
        # img = img.convert('RGB')
        img = F.to_tensor(img)
        img = F.normalize(img, img_mean, img_std)
        frames.append(img)
    frames = torch.stack(frames, dim=0)
    return frames.permute(1, 0, 2, 3)
