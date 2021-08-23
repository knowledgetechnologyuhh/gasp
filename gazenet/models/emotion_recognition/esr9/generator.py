import cv2
import torchvision.transforms.functional as F


def pre_process_input_image(img, img_width, img_height, img_mean, img_std):
    """
    Pre-processes an image for ESR-9.
    :param img: (ndarray)
    :return: (ndarray) image
    """

    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)
    img = F.normalize(img, img_mean, img_std)
    # img = transforms.Normalize(mean=ESR.INPUT_IMAGE_NORMALIZATION_MEAN, std=ESR.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(img)).unsqueeze(0)
    img = img.unsqueeze(0)
    return img