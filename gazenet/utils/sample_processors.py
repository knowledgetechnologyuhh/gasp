import threading
import pickle
import subprocess
import queue
import os
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import librosa

from gazenet.utils.helpers import stack_images

SERVER_MODE = False
REVIVE_RETRIES = 5

DEFAULT_SAMPLE_RATE = 16000

# audio capturer property ids
AUCAP_PROP_SAMPLE_RATE = 1001
AUCAP_PROP_CHUNK_SIZE = 1002
AUCAP_PROP_BUFFER_SIZE = 1003
AUCAP_PROP_CHANNELS = 1004
AUCAP_PROP_POS_FRAMES = 1005
AUCAP_PROP_FRAME_COUNT = 1007
# AUCAP_PROP_POS_MSEC = 1008


class SampleReader(object):
    """
    Handles the dataset reading process and provides a unified interface for
    """
    def __init__(self, pickle_file, mode=None, **kwargs):
        self.pickle_file = pickle_file
        self.samples = []
        self.video_id_to_sample_idx = {}
        self.len_frames = 0

        if pickle_file is not None:
            if mode == "r": # read
                self.read()

            elif mode == "w":  # write
                self.read_raw()
                self.write()
            elif mode == "x":  # safe write
                if os.path.exists(pickle_file):
                    raise FileExistsError("Read mode 'x' safely writes a file. "
                                  "Either delete the pickle_file or change the read mode")
                self.read_raw()
                self.write()
            elif mode == "a":  # append
                self.read()
                self.read_raw()
                self.write()
            elif mode == "d":  # dynamic: if the pickle_file exists it will be read, otherwise, a new dataset is created
                if os.path.exists(pickle_file):
                    self.read()
                else:
                    self.read_raw()
                    self.write()
            else:
                self.read_raw()
        else:
            if mode is not None:
                raise AttributeError("Specify the pickle_file attribute to make use of the mode")
            self.read_raw()

    def write(self):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump({"samples": self.samples,
                         "len_frames": self.len_frames,
                         "video_id_to_sample_idx": self.video_id_to_sample_idx,
                         "__name__": self.__class__.__name__}, f, pickle.HIGHEST_PROTOCOL)

    def read(self):
        with open(self.pickle_file, "rb") as f:
            data = pickle.load(f)
            assert data["__name__"] == self.__class__.__name__, \
                "The pickle_file has a mismatching name. " \
                "Ensure the correct pickle_file is read"

            self.len_frames = data["len_frames"]
            self.samples = data["samples"]
            self.video_id_to_sample_idx = data["video_id_to_sample_idx"]

    def read_raw(self):
        raise NotImplementedError("Not implemented for abstract Reader")

    def __len__(self):
        return len(self.samples)


class ImageCapture(object):
    """
    Loads all the image indices in a directory to the memory. When too many images are in a directory, use cv2.VideoCapture
    instead, making sure the images follow the string format provided and are ordered sequentially
    """
    def __init__(self, directory, extension="jpg", fps=1, sub_directories=False, image_file="captured_1", *args, **kwargs):
        self.properties = {
            cv2.CAP_PROP_POS_FRAMES: 0,
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
        if "." in source:  # file
            self.reader = self.__getfile__
            self.stream = librosa.load(source, sr=rate, duration=len_frames/buffer_size)
            self.frame_indices, chunk_size = np.linspace(0, len(self.stream[0]), num=len_frames, retstep=True, endpoint=False, dtype=int)
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
        frames = self.stream[0][curr_frame_idx:
                              curr_frame_idx + (self.get(AUCAP_PROP_BUFFER_SIZE)*self.get(AUCAP_PROP_CHUNK_SIZE))]
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


class SampleProcessor(object):
    """
    Handles video (images/audio) and processes it in a format suitable for render and display
    """
    def __init__(self, width=None, height=None, enable_audio=True,
                 video_reader=(cv2.VideoCapture,{}), audio_reader=(AudioCapture, {}), w_size=1, **kwargs):
        self.video_cap = None
        self.audio_cap = None
        self.width = width
        self.height = height
        self.enable_audio = enable_audio
        self.video_reader = video_reader
        self.audio_reader = audio_reader
        self.w_size = w_size
        self.video_out_path = ''
        self.audio_out_path = ''

        self.grabbed_video, self.video_frame, self.info, self.properties = False, None, {}, {}
        self.grabbed_audio,  self.audio_frames = False, None
        self.rt_index = 0
        self.started = False
        self.buffer = queue.Queue()
        self.read_lock = threading.Lock()
        self.annotation_properties = {}
        self.thread = None

    def load(self, metadata):
        # create the names for the output files
        self.video_out_path = os.path.dirname(os.path.join("gazenet", "readers", "visualization", "assets", "media", str(metadata["video_name"])))
        Path(self.video_out_path).mkdir(parents=True, exist_ok=True)
        self.video_out_path  = os.path.join(self.video_out_path, 'temp_vid_' + os.path.basename(str(metadata["video_name"])) + '.avi')
        if self.enable_audio and metadata["has_audio"]:
            self.audio_out_path = os.path.dirname(os.path.join("gazenet", "readers", "visualization", "assets", "media", str(metadata["audio_name"])))
            Path(self.audio_out_path).mkdir(parents=True, exist_ok=True)
            self.audio_out_path = os.path.join(self.audio_out_path, 'temp_aud_' + os.path.basename(str(metadata["video_name"])) + '.wav')

        if self.video_reader is not None:
            # setup the video capturer
            if self.video_cap is not None:
                self.video_cap.release()
            video_properties = self.video_reader[1].copy()
            self.video_cap = self.video_reader[0](metadata["video_name"], **video_properties)

        if self.enable_audio and metadata["has_audio"]:
            if self.audio_reader is not None:
                # setup the audio capturer
                if self.audio_cap is not None:
                    self.audio_cap.release()
                audio_properties = self.audio_reader[1].copy()
                if "buffer_size" not in audio_properties:
                    audio_properties["buffer_size"] = int(self.video_cap.get(cv2.CAP_PROP_FPS))
                if "len_frames" not in audio_properties:
                    audio_properties["len_frames"] = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # set to total num video_frames_list
                self.audio_cap = self.audio_reader[0](metadata["audio_name"], **audio_properties)

    def goto_frame(self, frame_index):
        if self.video_cap is not None:
            with self.read_lock:
                try:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                except:
                    pass
        if self.audio_cap is not None:
            with self.read_lock:
                try:
                    self.audio_cap.set(AUCAP_PROP_POS_FRAMES, int(frame_index))
                except:
                    pass

    def frames_per_sec(self):
        # if self.enable_audio and self.audio_cap is not None:
        #     # return audio frame for granular precision
        #     return self.audio_cap.get(AUCAP_PROP_FPS)
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_FPS))
        else:
            return 0

    def frame_index(self):
        # if self.enable_audio and self.audio_cap is not None:
        #     # return audio frame for granular precision
        #     return self.audio_cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            return 0

    def len_frames(self):
        # curr_sample = self.reader.samples[self.index]
        # return curr_sample['len_frames']
        if self.video_cap is not None:
            return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return 0

    def frame_size(self):
        if self.video_cap is not None:
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        else:
            return 0, 0

    def start(self, *args, **kwargs):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        with self.read_lock:
            self.started = True
        # start the video processing thread
        self.thread = threading.Thread(target=self.update, kwargs=kwargs, args=(self.buffer, *args))
        self.thread.start()
        return self

    def update(self, q, *args, **kwargs):
        retries = REVIVE_RETRIES
        while self.started:
            try:
                cmd = q.get(timeout=2)
                if cmd == 'play':
                    # TODO (fabawi): this loops over all extracted grouped_video_frames. they should be returned as a list instead
                    preprocessed_data = self.preprocess_frames(*args, **kwargs)
                    if preprocessed_data is not None:
                        extracted_data_list = self.extract_frames(**preprocessed_data)
                    else:
                        extracted_data_list = self.extract_frames()

                    for extracted_data in zip(*extracted_data_list):
                        if self.annotation_properties:
                            grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = \
                                self.annotate_frame(extracted_data, *args, **self.annotation_properties)
                        else:
                            grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = \
                                self.annotate_frame(extracted_data, *args, **kwargs)

                        if grabbed_video:
                            if self.width is not None and self.height is not None:
                                video_frame = cv2.resize(stack_images(grouped_video_frames), (self.width, self.height))
                            else:
                                video_frame = stack_images(grouped_video_frames)
                        else:
                            video_frame = None

                        if grabbed_audio:
                            audio_frames = audio_frames.flatten()
                        else:
                            audio_frames = None

                        with self.read_lock:
                            self.grabbed_video = grabbed_video
                            self.video_frame = video_frame
                            self.info = info
                            self.properties = properties
                            if self.enable_audio:
                                self.grabbed_audio = grabbed_audio
                                if self.grabbed_audio:
                                    self.audio_frames = audio_frames

                elif cmd == 'pause':
                    # if self.audio_cap is not None:
                    #     with self.read_lock:
                    #         self.audio_cap["curr_frame"] = self.frame_index()
                    continue
                elif cmd == 'stop':
                    with self.read_lock:
                        self.started = False
            except queue.Empty:
                retries -= 1
                if retries == 0:
                    with self.read_lock:
                        self.started = False
                    break
                continue

    def read(self):
        grabbed_video = self.grabbed_video
        if self.video_frame is not None:
            video_frame = self.video_frame.copy()
        else:
            video_frame = self.video_frame
        info = self.info.copy()
        properties = self.properties.copy()

        grabbed_audio = self.grabbed_audio
        audio_frames = self.audio_frames

        return grabbed_video, video_frame, grabbed_audio, audio_frames, info, properties

    def stop(self):
        if SERVER_MODE:
            if self.video_cap is not None:
                self.video_cap.release()
            if self.audio_cap is not None:
                self.audio_cap.release()
        with self.read_lock:
            self.started = False
        if self.thread is not None:
            self.thread.join()
        print("Stopped the capture thread")

    def __exit__(self, exec_type, exc_value, traceback):
        if self.video_cap is not None:
            self.video_cap.release()
        if self.audio_cap is not None:
            self.audio_cap.release()

    def play(self):
        with self.read_lock:
            self.buffer.put('play')

    def pause(self):
        with self.read_lock:
            self.buffer.put('pause')

    def preprocess_frames(self, *args, **kwargs):
        return None

    def postprocess_frames(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list,
                           info_list, properties_list,
                           keep_video=True, keep_audio=True, keep_info=True, keep_properties=True,
                           info_override=None, properties_override=None, plot_override=None,
                           keep_plot_frames_only=False, resize_frames=False, convert_plots_gray=False,
                           duplicate_audio_frames=False,
                           *args, **kwargs):
        # TODO (fabawi): these may break some functionality. Make sure to externally handle None values
        if not keep_video:
            grouped_video_frames_list = [None] * len(grouped_video_frames_list)
        if not keep_audio:
            audio_frames_list = [None] * len(audio_frames_list)
        if not keep_properties:
            properties_list = [None] * len(properties_list)
        if not keep_info:
            info_list = [None] * len(info_list)

        for grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties in zip(
                grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list,info_list, properties_list):

            if keep_video:
                if plot_override is not None:
                    grouped_video_frames["PLOT"] = plot_override
                if keep_plot_frames_only or resize_frames or convert_plots_gray:
                    del_frame_names = []
                    keep_plot_names = [item for sublist in grouped_video_frames["PLOT"] for item in sublist] + ["PLOT"]
                    for plot_name, plot in grouped_video_frames.items():
                        if keep_plot_frames_only:
                            if plot_name not in keep_plot_names:
                                grouped_video_frames[plot_name] = None
                                del_frame_names.append(plot_name)
                        if plot_name != "PLOT" and plot is not None:
                            if resize_frames:
                                if grabbed_video:
                                    grouped_video_frames[plot_name] = cv2.resize(plot.copy(), (self.width, self.height))
                                # else:
                                #     grouped_video_frames[plot_name] = np.zeros((self.height, self.width, 3))
                            if convert_plots_gray:
                                if grabbed_video:
                                    grouped_video_frames[plot_name] = cv2.cvtColor(plot.copy(), cv2.COLOR_RGB2GRAY)
                                # else:
                                #     grouped_video_frames[plot_name] = np.zeros((self.height, self.width, 1))

                    for del_frame_name in del_frame_names:
                        del grouped_video_frames[del_frame_name]

            if keep_audio:
                if duplicate_audio_frames:
                    raise NotImplementedError

            if keep_info:
                if info_override is not None:
                    # overrides names in list at the surface level of the dictionary
                    for override in info_override:
                        try:
                            del info[override]
                        except:
                            pass
            if keep_properties:
                if properties_override is not None:
                    # overrides names in list at the surface level of the dictionary
                    for override in properties_override:
                        try:
                            del properties[override]
                        except:
                            pass

        return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, \
               info_list, properties_list

    def extract_frames(self, *args, extract_video=True, extract_audio=True, realtime_indexing=False, **kwargs):
        grabbed_video_list = []
        grouped_video_frames_list = []
        audio_frames_list = []
        grabbed_audio_list = []
        info_list = []
        properties_list = []
        for w_idx in range(self.w_size):
            if self.video_cap.isOpened() and extract_video:
                grabbed_video, video_frame = self.video_cap.read()
            else:
                grabbed_video, video_frame = False, None
            grouped_video_frames_list.append({"captured": video_frame})
            grabbed_video_list.append(grabbed_video)
            if self.enable_audio and self.audio_cap.isOpened() and extract_audio:
                grabbed_audio, audio_frames = self.audio_cap.read(*args, stateful=True, **kwargs)
            else:
                grabbed_audio, audio_frames = False, None
            audio_frames_list.append(audio_frames)
            grabbed_audio_list.append(grabbed_audio)
            if realtime_indexing:
                info_list.append({"frame_info": {"frame_id": self.rt_index}})
                with self.read_lock:
                    self.rt_index += 1
            else:
                info_list.append({"frame_info": {"frame_id": self.frame_index()}})
            properties_list.append({})

        return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list

    def annotate_frames(self, input_data_list, plotter, *args, **kwargs):
        grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list = [], [], [], []
        info_list, properties_list = [], []
        for extracted_data in zip(*input_data_list):
            if self.annotation_properties:
                grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = \
                    self.annotate_frame(extracted_data, plotter, *args, **self.annotation_properties)
            else:
                grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = \
                    self.annotate_frame(extracted_data, plotter, *args, **kwargs)
            grabbed_video_list.append(grabbed_video)
            grouped_video_frames_list.append(grouped_video_frames)
            grabbed_audio_list.append(grabbed_audio)
            audio_frames_list.append(audio_frames)
            info_list.append(info)
            properties_list.append(properties)
        return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list

    def annotate_frame(self, input_data, plotter, *args, **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        grouped_video_frames = {"PLOT": [["captured"]], **grouped_video_frames}
        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def set_annotation_properties(self, annotation_properties):
        with self.read_lock:
            self.annotation_properties = annotation_properties

    # def _write_audio_video(self):
    #     # write the audio to a file
    #     if self.audio_writer is not None:
    #         self.video_writer.release()
    #         self.audio_writer.release()
    #         # if self.enable_audio and self.audio_cap is not None:
    #         # librosa.output.write_wav(self.audio_out_path, *self.audio_cap['audio'])
    #         cmd = 'ffmpeg -y -i ' + \
    #               self.audio_out_path + '  -r ' + \
    #               str(self.frames_per_sec()) + ' -i ' + \
    #               self.video_out_path + '  -filter:a aresample=async=1 -c:a flac -c:v copy ' + \
    #               self.video_out_path + '.mkv'
    #         subprocess.call(cmd, shell=True)  # "Muxing Done
    #         print('Muxing done')
    #     elif self.video_writer is not None:
    #         self.video_writer.release()


class InferenceSampleProcessor(SampleProcessor):
    """
    Wraps inference classes to support integration into visualizers and include most functionality
    supported by the video processor
    """
    def __init__(self, width=None, height=None, w_size=1, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, enable_audio=False, video_reader=None, audio_reader=None,
                         **kwargs)

    def infer_frame(self, *args, **kwargs):
        raise NotImplementedError("Infer not defined in base class")

    def extract_frames(self, *args, **kwargs):
        return self.infer_frame(*args, **kwargs)

    def preprocess_frames(self, grabbed_video_list, grouped_video_frames_list,
                          grabbed_audio_list, audio_frames_list, info_list, properties_list, **kwargs):
        if kwargs:
            features = {**kwargs}
        else:
            features = {}
        grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, \
        info_list, properties_list = list(grabbed_video_list), list(grouped_video_frames_list), \
                                     list(grabbed_audio_list), list(audio_frames_list), \
                                     list(info_list), list(properties_list)
        pad = 1
        if self.w_size > len(grabbed_video_list):
            pad = self.w_size + 1 - len(grabbed_video_list)
        lim = min(self.w_size - 1, len(grabbed_video_list) - 1)
        features["preproc_pad_len"] = pad
        features["preproc_lim_len"] = lim
        features["grabbed_video_list"] = grabbed_video_list[:lim] + [grabbed_video_list[lim]] * pad
        features["grouped_video_frames_list"] = grouped_video_frames_list[:lim] + [grouped_video_frames_list[lim]] * pad

        if grabbed_audio_list:
            aud_lim = min(lim, len(grabbed_audio_list) - 1)
            aud_pad = pad - (aud_lim-lim)
            features["grabbed_audio_list"] = grabbed_audio_list[:aud_lim] + [grabbed_audio_list[aud_lim]] * aud_pad
            features["audio_frames_list"] = audio_frames_list[:aud_lim] + [audio_frames_list[aud_lim]] * aud_pad
        else:
            features["grabbed_audio_list"] = grabbed_audio_list * pad
            features["audio_frames_list"] = audio_frames_list * pad

        features["info_list"] = info_list[:lim] + [info_list[lim]] * pad
        features["properties_list"] = properties_list[:lim] + [properties_list[lim]] * pad

        return features

    def _keep_extracted_frames_data(self, source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                    grabbed_audio_list, audio_frames_list, info_list, properties_list):
        if source_frames_idxs is not None:
            grabbed_video_list = [grabbed_video_list[i] for i in source_frames_idxs]
            grouped_video_frames_list = [grouped_video_frames_list[i] for i in source_frames_idxs]
            if grabbed_audio_list:
                grabbed_audio_list = [grabbed_audio_list[i] for i in source_frames_idxs]
                audio_frames_list = [audio_frames_list[min(len(audio_frames_list)-1,i)] for i in source_frames_idxs]
            else:
                grabbed_audio_list = [[]] * len(source_frames_idxs)
                audio_frames_list = [[]] * len(source_frames_idxs)
            info_list = [info_list[i] for i in source_frames_idxs]
            properties_list = [properties_list[i] for i in source_frames_idxs]
            return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, \
                   info_list, properties_list
        else:
            return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, \
                   info_list, properties_list
