import threading

import cv2
import numpy as np
import sounddevice as sd
import librosa

from gazenet.utils.registrar import *
from gazenet.utils.constants import *

DEFAULT_SAMPLE_RATE = 16000


@VideoCaptureRegistrar.register
class VideoCapture(cv2.VideoCapture):
    def __init__(self, *args, fps=None, **kwargs):
        super().__init__(*args, **kwargs)


@VideoCaptureRegistrar.register
class ImageCapture(object):
    """
    Loads all the image indices in a directory to the memory. When too many images are in a directory, use cv2.VideoCapture
    instead, making sure the images follow the string format provided and are ordered sequentially
    """
    def __init__(self, directory, extension="jpg", fps=0, sub_directories=False, image_file="captured_1", *args, **kwargs):
        self.properties = {
            cv2.CAP_PROP_POS_FRAMES: 0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: None,
            cv2.CAP_PROP_FRAME_WIDTH: None,
            cv2.CAP_PROP_FRAME_HEIGHT: None
        }
        self.directory = directory
        self.sub_directories = True

        if sub_directories:
            self.frames = [os.path.join(f, image_file+"."+extension) for f in os.listdir(directory)]
            self.frames = sorted(self.frames, key=lambda x: float(x.split(os.sep, 1)[0]))
        else:
            self.frames = [f for f in os.listdir(directory) if f.endswith("."+extension)]
            self.frames = sorted(self.frames, key=lambda x: float(x[:-(len(extension)+1)]))
        self.set(cv2.CAP_PROP_FRAME_COUNT, len(self.frames))

        file = os.path.join(self.directory, self.frames[0])
        im = cv2.imread(file)
        h, w, c = im.shape
        self.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.opened = True

    def read(self):
        try:
            frame_index = self.get(cv2.CAP_PROP_POS_FRAMES)
            file = os.path.join(self.directory, self.frames[frame_index])
            im = cv2.imread(file)
            self.opened = True
            self.set(cv2.CAP_PROP_POS_FRAMES, frame_index+1)
            self.set(cv2.CAP_PROP_POS_MSEC, self.get(cv2.CAP_PROP_POS_FRAMES) * (1/self.get(cv2.CAP_PROP_FPS)))
            return True, im
        except:
            self.opened = False
            return False, None

    def isOpened(self):
            return self.opened

    def release(self):
        self.frames = []

    def set(self, propId, value):
        self.properties[propId] = value

    def get(self, propId):
        return self.properties[propId]


@AudioCaptureRegistrar.register
class AudioCapture(object):
    """
    Loads an entire audio file into memory (no buffering due to limited format support) and microphone (blocking)
    """
    def __init__(self, source, buffer_size=30, rate=None, channels=1, len_frames=1):
        self.properties = {
            AUCAP_PROP_BUFFER_SIZE: buffer_size,
            AUCAP_PROP_SAMPLE_RATE: rate,
            AUCAP_PROP_CHANNELS: channels,
            AUCAP_PROP_CHUNK_SIZE: None,
            AUCAP_PROP_FRAME_COUNT: len_frames,
            AUCAP_PROP_POS_FRAMES: 0,
        }

        # capturing mode
        if isinstance(source, str) and "." in source:  # file
            self.reader = self.__getfile__
            self.stream = librosa.load(source, sr=rate, duration=len_frames/buffer_size, mono=True if channels == 1 else False)
            if len(self.stream[0].shape) != 1:
                stream_len = len(self.stream[0][-1])
            else:
                stream_len = len(self.stream[0])
            self.frame_indices, chunk_size = np.linspace(0, stream_len, num=len_frames, retstep=True, endpoint=False, dtype=int, axis=-1)
            self.set(AUCAP_PROP_CHUNK_SIZE, int(chunk_size))
            self.set(AUCAP_PROP_SAMPLE_RATE, self.stream[1])

        else:  # microphone
            # TODO (fabawi): microphone reading is very choppy
            self.reader = self.__getmic__
            if len_frames <= 0:
                len_frames = 1
            if rate is None:
                device_info = sd.query_devices(source, 'input')
                rate = int(device_info['default_samplerate'])
                self.set(AUCAP_PROP_SAMPLE_RATE, int(rate))
            chunk_size = int(rate * len_frames / buffer_size)
            self.stream = sd.InputStream(device=source,
                                         samplerate=rate,
                                         channels=channels,
                                         blocksize=chunk_size*buffer_size)
            self.stream.start()
            self.set(AUCAP_PROP_CHUNK_SIZE, chunk_size)
        self.opened = True
        self.state = -1
        self.read_lock = threading.Lock()

    def __getmic__(self):
        frames = self.stream.read(self.get(AUCAP_PROP_CHUNK_SIZE)*self.get(AUCAP_PROP_BUFFER_SIZE))
        frames = np.array(np.split(frames[0], self.get(AUCAP_PROP_BUFFER_SIZE)))
        return frames
        pass

    def __getfile__(self):
        curr_frame_idx = self.frame_indices[self.get(AUCAP_PROP_POS_FRAMES)]
        if len(self.stream[0].shape) != 1:
            frames = self.stream[0][:, curr_frame_idx: curr_frame_idx + (self.get(AUCAP_PROP_BUFFER_SIZE) *
                                                                         self.get(AUCAP_PROP_CHUNK_SIZE))]
            frames = np.array(np.moveaxis(np.split(frames, self.get(AUCAP_PROP_BUFFER_SIZE), axis=-1), 1, 0))
        else:
            frames = self.stream[0][curr_frame_idx: curr_frame_idx + (self.get(AUCAP_PROP_BUFFER_SIZE) *
                                                                      self.get(AUCAP_PROP_CHUNK_SIZE))]
            frames = np.array(np.split(frames, self.get(AUCAP_PROP_BUFFER_SIZE)))

        return frames

    def read(self, *args, stateful=False, **kwargs):
        # TODO (fabawi): being stateful causes issues with seeking. Also, the buffer size should be larger to avoid
        #  pauses (should be dynamic and loads the whole clip when reading from disk)
        try:
            if stateful:
                with self.read_lock:
                    self.state += 1
                    self.state %= self.get(AUCAP_PROP_BUFFER_SIZE)
                if self.state != 0:
                    return False, None

            frame = self.reader()
            with self.read_lock:
                self.set(AUCAP_PROP_POS_FRAMES,
                         self.get(AUCAP_PROP_POS_FRAMES) + self.get(AUCAP_PROP_BUFFER_SIZE))
                self.opened = True
            return True, frame
        except:
            with self.read_lock:
                self.opened = False
            return False, None

    def isOpened(self):
            return self.opened

    def release(self):
        try:
            self.stream.stop()
        except:
            self.stream = None
        with self.read_lock:
            self.state = 0

    def set(self, propId, value):
        self.properties[propId] = value

    def get(self, propId):
        return self.properties[propId]
