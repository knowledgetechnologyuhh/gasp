import threading
import subprocess
import queue
import os
from pathlib import Path

import cv2

from gazenet.utils.registrar import *
from gazenet.utils.helpers import stack_images
from gazenet.utils.constants import *

try:  # pickle with protocol 5 if python<3.8
    import pickle5 as pickle
except:
    import pickle

SERVER_MODE = False
REVIVE_RETRIES = 5
DEFAULT_FPS = 25


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
            if self.__class__.__name__ != "SampleReader":
                assert data["__name__"] == self.__class__.__name__, \
                    "The pickle_file has a mismatching name. " \
                    "Ensure the correct pickle_file is read"

            self.len_frames = data["len_frames"]
            self.samples = data["samples"]
            self.video_id_to_sample_idx = data["video_id_to_sample_idx"]

    def read_raw(self):
        raise NotImplementedError("Not implemented for base Reader")

    def __len__(self):
        return len(self.samples)


class SampleProcessor(object):
    """
    Handles video (images/audio) and processes it in a format suitable for render and display
    """
    def __init__(self, width=None, height=None, enable_audio=True,
                 video_reader=("VideoCapture", {}), audio_reader=("AudioCapture", {}), w_size=1, sample_stride=1,
                 # sample_overlap=False,  # sample_overlap is always False because reading is iterative
                 **kwargs):
        self.video_cap = None
        self.audio_cap = None
        self.width = width
        self.height = height
        self.enable_audio = enable_audio

        if video_reader is not None and isinstance(video_reader[0], str):
            VideoCaptureRegistrar.scan()
            self.video_reader = (VideoCaptureRegistrar.registry[video_reader[0]], video_reader[1])
        else:
            self.video_reader = video_reader

        if audio_reader is not None and isinstance(audio_reader[0], str):
            AudioCaptureRegistrar.scan()
            self.audio_reader = (AudioCaptureRegistrar.registry[audio_reader[0]], audio_reader[1])
        else:
            self.audio_reader = audio_reader

        self.w_size = w_size
        self.sample_stride = sample_stride

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
        # self.video_out_path = os.path.dirname(os.path.join("gazenet", "readers", "visualization", "assets", "media", str(metadata["video_name"])))
        # Path(self.video_out_path).mkdir(parents=True, exist_ok=True)
        # self.video_out_path  = os.path.join(self.video_out_path, 'temp_vid_' + os.path.basename(str(metadata["video_name"])) + '.avi')
        # if self.enable_audio and metadata["has_audio"]:
        #     self.audio_out_path = os.path.dirname(os.path.join("gazenet", "readers", "visualization", "assets", "media", str(metadata["audio_name"])))
        #     Path(self.audio_out_path).mkdir(parents=True, exist_ok=True)
        #     self.audio_out_path = os.path.join(self.audio_out_path, 'temp_aud_' + os.path.basename(str(metadata["video_name"])) + '.wav')

        if self.video_reader is not None:
            # setup the video capturer
            if self.video_cap is not None:
                self.video_cap.release()
            video_properties = self.video_reader[1].copy()
            metadata["video_name"] = video_properties.get("video_name", metadata["video_name"])
            video_properties.pop("video_name", None)
            video_properties.update(fps=metadata.get("video_fps", DEFAULT_FPS))
            self.video_cap = self.video_reader[0](metadata["video_name"], **video_properties)

        if self.enable_audio and metadata["has_audio"]:
            if self.audio_reader is not None:
                # setup the audio capturer
                if self.audio_cap is not None:
                    self.audio_cap.release()
                audio_properties = self.audio_reader[1].copy()
                metadata["audio_name"] = audio_properties.get("audio_name", metadata["audio_name"])
                audio_properties.pop('audio_name', None)
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

    def frame_time(self):
        if self.video_cap is not None:
            return float(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            return 0.0

    def len_frames(self):
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
                            if len(audio_frames.shape) == 2:
                                audio_frames = audio_frames.flatten()
                            else:
                                audio_frames = audio_frames.reshape(-1, audio_frames.shape[-1]*audio_frames.shape[-2])
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
                for sample_stride in range(self.sample_stride):
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
        # TODO (fabawi): we disable audio for now, but might want to change that if we want to store
        #  the captured audio on detection. In general, a better pattern needs to be followed for this to work
        super().__init__(width=width, height=height, w_size=w_size, enable_audio=False, video_reader=None, audio_reader=None,
                         **kwargs)

    def infer_frame(self, *args, **kwargs):
        raise NotImplementedError("Not implemented for base Inferer")

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


class ApplicationProcessor(object):
    """
    Every application must inherit this class. Applications are flexible in how they execute, but are limited in what
    they can return for further processing
    """
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Not implemented for base Application")
