import time
import queue
import copy
import threading

import numpy as np
import cv2
import sounddevice as sd
from scipy.spatial.distance import cdist
from gazenet.utils.registrar import *
from gazenet.utils.annotation_plotter import OpenCV
import gazenet.utils.sample_processors as sp

cv2.ocl.setUseOpenCL(False)

read_lock = threading.Lock()


def view(plotter, video, n_frames=5, enable_audio=True, video_properties={}):
    video.start(plotter=plotter, **video_properties)
    # audio_streamer = sd.play(sample_streamer.audio_cap["audio"][0]
    #                          [sample_streamer.audio_cap["video_frames_list"][sample_streamer.audio_cap["curr_frame"]]:],
    #                     sample_streamer.audio_cap["audio"][1])
    i = 0
    buff_audio_frames = []
    try:
        while i < n_frames:
            ts = time.time()
            # if i < 100:
            #     sample_streamer.play()
            # elif i < 200:
            #     sample_streamer.pause()
            # elif i == 200:
            #     sample_streamer.goto_frame(0)
            #     sample_streamer.pause()
            # else:
            #     sample_streamer.play()
            video.play()

            grabbed_video, video_frame, grabbed_audio, audio_frames, _, _ = video.read()
            if enable_audio:
                if len(buff_audio_frames) >= video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE):
                    with read_lock:
                        new_buff_audio_frames = buff_audio_frames.copy()
                        buff_audio_frames = []
                    # sd.wait()
                    sd.play(np.array(new_buff_audio_frames), video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE), mapping=[1,2])
                if grabbed_audio:
                    with read_lock:
                        buff_audio_frames.extend(audio_frames)
            if grabbed_video:
                cv2.imshow('Frame', video_frame)
                orig_fps = video.frames_per_sec() * 1.06  # used to be multiplied by 1.51
                td = time.time() - ts
                if td < 1.0/orig_fps:
                    cv2.waitKey(int((1.0/orig_fps - td)*1000)) & 0xFF
                else:
                    cv2.waitKey(1) & 0xFF
            else:
                raise Exception
            i += 1
        video.stop()
        # sd.stop()
        cv2.destroyAllWindows()
    except:
        video.stop()
        # sd.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    width, height = 1600, 800
    play_audio = True

    # define the reader
    reader = "FindWhoSampleReader"
    sampler = "FindWhoSample"
    # sampler_properties = {"show_saliency_map": True, "enable_transform_overlays":False, "color_map": "bone"}
    sampler_properties = {"show_saliency_map": False, "participant": 1}


    SampleRegistrar.scan()
    ReaderRegistrar.scan()

    plotter = OpenCV()
    video_source = ReaderRegistrar.registry[reader](mode="d")
    video = SampleRegistrar.registry[sampler](video_source, w_size=1, width=width, height=height, enable_audio=play_audio, audio_reader=("AudioCapture", {"channels": 1}))
    print("total videos", len(video))
    for i in range(len(video)*5):
        vid_id = i%len(video)
        print("video", i+1)
        view(plotter, video, n_frames=2000, enable_audio=play_audio, video_properties=sampler_properties)
        next(video)