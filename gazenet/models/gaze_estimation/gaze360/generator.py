import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image

def pre_process_input_image(img, img_width, img_height, img_mean, img_std):
    """
    Pre-processes an image for gaze360.
    :param img: (ndarray)
    :return: (ndarray) image
    """

    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)
    img = F.normalize(img, img_mean, img_std)
    # img = transforms.Normalize(mean=ESR.INPUT_IMAGE_NORMALIZATION_MEAN, std=ESR.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(img)).unsqueeze(0)
    return img


def spherical_to_compatible_form(tpr):
    ptr = torch.zeros(tpr.size(0), 3)
    ptr[:, 0] = tpr[:, 1]
    ptr[:, 1] = tpr[:, 0]
    ptr[:, 2] = 1
    return ptr


def spherical_to_cartesian(tpr):
    xyz = torch.zeros(tpr.size(0),3)
    xyz[:,2] = -torch.cos(tpr[:,1])*torch.cos(tpr[:,0])
    xyz[:,0] = torch.cos(tpr[:,1])*torch.sin(tpr[:,0])
    xyz[:,1] = torch.sin(tpr[:,1])
    return xyz