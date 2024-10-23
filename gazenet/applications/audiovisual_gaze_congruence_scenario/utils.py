import datetime

import cv2
import pandas as pd
import numpy as np

# based on: https://github.com/fabawi/ros_dash_video_stream/blob/master/fps.py
class FPS(object):
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start and reset the timer
        self._start = datetime.datetime.now()
        self._numFrames = 0
        self._end = None
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()


    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


# based on: https://github.com/fabawi/ros_dash_video_stream/blob/master/video_streamers.py
class VideoSource(object):
    """ The super class which video resources must inherit"""
    def __init__(self, image_size=(1080, 1920)):
        self.image_size = image_size
        self.__len = image_size[0] * image_size[1] * 3

    def isOpened(self):
        return True

    def read(self):
        rval = True
        frame = np.random.randint(0, 255,  (self.image_size[0],self.image_size[1], 3), np.uint8)
        return rval, frame

    def __len__(self):
        return self.__len

    def __str__(self):
        return 'Video Source'


# based on: https://github.com/fabawi/ros_dash_video_stream/blob/master/video_streamers.py
class VideoStreamer(object):
    def __init__(self):
        self.fps = FPS().start()

    def get_fps(self):
        self.fps.stop()
        fps = self.fps.fps()
        self.fps.start()
        return fps

    def get_frame(self):
        raise NotImplementedError("Choose a video streamer from the available ones "
                                  "e.g., CV2VideoStreamer or ROSVideoStreamer")


# based on: https://github.com/fabawi/ros_dash_video_stream/blob/master/video_streamers.py
class CV2VideoStreamer(VideoStreamer):
    def __init__(self, resource=None):
        super(CV2VideoStreamer, self).__init__()
        if resource is None:
            print("You must give an argument to open a video stream.")
            print("  It can be a number as video device,  e.g.: 0 would be /dev/video0")
            print("  It can be a url of a stream,         e.g.: rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
            print("  It can be a video file,              e.g.: samples/moon.avi")
            print("  It can be a class generating images, e.g.: TimeStampVideo")
            exit(0)

        # If given a number interpret it as a video device
        if isinstance(resource, int) or len(resource) < 3:
            self.resource_name = "/dev/video" + str(resource)
            resource = int(resource)
            self.vidfile = False
        else:
            self.resource_name = str(resource)
            self.vidfile = True
        print("Trying to open resource: " + self.resource_name)

        if isinstance(resource, VideoSource):
            self.cap = resource
        else:
            self.cap = cv2.VideoCapture(resource)

        if not self.cap.isOpened():
            print("Error opening resource: " + str(resource))
            exit(0)
        self.fps = FPS().start()

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, image = self.cap.read()
        # TODO (fabawi): resizing the image takes some time. make it multi-threaded
        # image = imutils.resize(image, width=VID_WIDTH)

        ret, jpeg = cv2.imencode('.jpg', image)
        jpeg_bytes = jpeg.tobytes()
        self.fps.update()
        return jpeg_bytes
