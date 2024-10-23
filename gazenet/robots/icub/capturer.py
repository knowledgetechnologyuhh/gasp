import argparse
import time

import cv2
import numpy as np

try:
    from wrapyfi.connect.wrapper import MiddlewareCommunicator
    import yarp
except:
    print("Install YARP and Wrapyfi to capture stimuli using the iCub robot")

from gazenet.utils.registrar import *


@VideoCaptureRegistrar.register
class ICubVideoCapture(MiddlewareCommunicator):
    """
    Loads all the image indices in a directory to the memory. When too many images are in a directory, use cv2.VideoCapture
    instead, making sure the images follow the string format provided and are ordered sequentially
    """
    def __init__(self, camera="/icub/cam/left", fps=30, *args, **kwargs):
        super(MiddlewareCommunicator, self).__init__()

        self.properties = {
            cv2.CAP_PROP_POS_FRAMES: 0,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FRAME_WIDTH: None,
            cv2.CAP_PROP_FRAME_HEIGHT: None
        }

        self.cam_props = {"port_cam": camera}

        # control the listening properties from within the app
        self.activate_communication("receive_images", "listen")

        self.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.opened = True

    @MiddlewareCommunicator.register("Image", "yarp", "ICubVideoCapture", "$port_cam", carrier="", width=320, height=240, rgb=True)
    def receive_images(self, port_cam):
        robo_cam = None
        return robo_cam,

    def read(self):
        try:
            # realtime capture with no fps interface, so we need to wait according to fps
            fps = self.get(cv2.CAP_PROP_FPS)
            time.sleep(1/fps)

            frame_index = self.get(cv2.CAP_PROP_POS_FRAMES)
            im, = self.receive_images(**self.cam_props)
            self.opened = True
            self.set(cv2.CAP_PROP_POS_FRAMES, frame_index+1)
            return True, im
        except:
            self.opened = False
            return False, None

    def isOpened(self):
            return self.opened

    def release(self):
        pass

    def set(self, propId, value):
        self.properties[propId] = value

    def get(self, propId):
        return self.properties[propId]


@VideoCaptureRegistrar.register
class ICubSimpleVideoCapture(MiddlewareCommunicator):
    """
    Loads all the image indices in a directory to the memory. When too many images are in a directory, use cv2.VideoCapture
    instead, making sure the images follow the string format provided and are ordered sequentially
    """
    def __init__(self):
        super(MiddlewareCommunicator, self).__init__()
        self.activate_communication("receive_sim_images", "listen")

    @MiddlewareCommunicator.register("Image", "yarp", "ICubSimpleVideoCapture", "$port_cam", carrier="", width=320, height=240, rgb=True)
    def receive_sim_images(self, port_cam):
        robo_cam = None
        return robo_cam,

    def get_left_eye_image(self):
        """
        Get an image of what the iCub's left eye is currently seeing
        @return: The image
        """
        return self.get_eye_image("left")

    def get_right_eye_image(self):
        """
        Get an image of what the iCub's right eye is currently seeing
        @return: The image
        """
        return self.get_eye_image("right")

    def get_eye_image(self, eye="left"):
        """
        Get an image of what the iCub's eye is currently seeing
        @return: The image
        """
        img = np.zeros(1)
        retries = 0
        while img is None or img.max(initial=0) == 0:
            img, = self.receive_sim_images(port_cam=f"/icubSim/cam/{eye}")
            retries += 1
            if retries > 20:
                raise RuntimeError("Failed to receive image from simulator")
        return img


