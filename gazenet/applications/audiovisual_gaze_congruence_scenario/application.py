#
# Run the simulated gaze videos
#
import threading
from threading import Thread
import random
import time
import argparse

import cv2
import sounddevice as sd
import pandas as pd
from joblib import Parallel, delayed


from gazenet.utils.annotation_plotter import OpenCV
import gazenet.utils.sample_processors as sp
from gazenet.utils.sample_processors import ApplicationProcessor
from gazenet.utils.registrar import *

try:
    import gazenet.applications.audiovisual_gaze_congruence_scenario.dash_visualizer as dash_app
    from gazenet.applications.audiovisual_gaze_congruence_scenario.utils import *
except:
    print("Install PLOTY-DASH to enable triggering HRI viewer server mode")

try:
    from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
except:
    print("Install Wrapyfi to enable triggering HRI video player. "
          "Make sure WRAPYFI_DEFAULT_COMMUNICATOR environment variable is set "
          "to whichever middleware you are using (e.g. yarp). If `pip install wrapyfi[all]` was used to install Wrapyfi, "
          "then ZeroMQ is installed and the default communicator is automatically set to ZeroMQ.")

try:
    import keyboard
except:
    print("Install keyboard to enable keyboard capture during HRI experiments. "
          "NOTE: Keyboard can only run as a superuser")


cv2.ocl.setUseOpenCL(False)

read_lock = threading.Lock()

def run_webapp():
    dash_app.app.run_server(host='127.0.0.1', port=8050, debug=False, use_reloader=False)


class HRIMoviePlayer(MiddlewareCommunicator):
    def __init__(self):
        super(MiddlewareCommunicator, self).__init__()
        # monkey patching since we don't want to import these modules when the specific app is not called
        from moviepy.editor import VideoFileClip
        import pygame
        self._VideoFileClip = VideoFileClip
        self._pygame = pygame
        # self._pygame.init()
        # self._pygame.mixer.quit()

    def capture_key(self, keys):
        while True:
            for key in keys:
                if keyboard.is_pressed(key):
                    return key

    def play_video(self, video_file, delay=1.2, fullscreen=True):
        time.sleep(delay)
        # pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        clip = self._VideoFileClip(video_file)
        clip.preview(fullscreen=fullscreen)
        self._pygame.quit()

    def play_video_capture(self, plotter, video, n_frames=5, enable_audio=True, video_properties={}):
        video.start(plotter=plotter, **video_properties)
        i = 0
        buff_audio_frames = []
        try:
            while i < n_frames:
                ts = time.time()
                video.play()

                grabbed_video, video_frame, grabbed_audio, audio_frames, _, _ = video.read()
                if enable_audio:
                    if len(buff_audio_frames) >= video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE):
                        with read_lock:
                            new_buff_audio_frames = buff_audio_frames.copy()
                            buff_audio_frames = []
                        # sd.wait()
                        sd.play(np.array(new_buff_audio_frames), video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE),
                                mapping=[1, 2])
                    if grabbed_audio:
                        with read_lock:
                            buff_audio_frames.extend(audio_frames)
                if grabbed_video:
                    cv2.imshow('Frame', video_frame)
                    orig_fps = video.frames_per_sec() * 1.06  # used to be multiplied by 1.51
                    td = time.time() - ts
                    if td < 1.0 / orig_fps:
                        cv2.waitKey(int((1.0 / orig_fps - td) * 1000)) & 0xFF
                    else:
                        cv2.waitKey(1) & 0xFF
                else:
                    raise Exception
                i += 1
            video.stop()
            cv2.destroyAllWindows()
        except:
            video.stop()
            cv2.destroyAllWindows()

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "HRIMoviePlayer", "/movieplayer/hri_gaze_scenario", carrier="")
    def exchange_object(self, participant_idx, condition_congruency, auditory_localization, visualcue_localization):
        obj = {"participant_idx": participant_idx,
               "conditions_congruency": condition_congruency,
               "auditory_localization": auditory_localization,
               "visualcue_localization": visualcue_localization}
        return obj,


@ApplicationRegistrar.register
class PlayHRIVideoApplication(ApplicationProcessor):

    def __init__(self, width=None, height=None, w_size=1, condition="random", **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, condition=condition, **kwargs)
        self.short_name = "playhrivideo"

        self.condition = condition

        if "csv" in self.condition:
            self.participant_conditions = pd.read_csv(condition)
        else:
            self.participant_conditions = None

        self.pointer = 0
        self.movie_player = HRIMoviePlayer()
        self.movie_player.activate_communication("exchange_object", mode="publish")

    def execute(self, previous_data=None, dummy_prop=False,
                       **kwargs):
        if previous_data and previous_data is not None:
            _, _, _, _, info, properties = previous_data
        else:
            properties = {}
            info = {}

        # randomly create condition for video
        if self.condition == "random" or self.participant_conditions is None or self.participant_conditions.iloc[self.pointer]["participant_idx"] == 0:
            aloc = random.choice(["left", "right"])
            vloc = random.choice([random.choice(["left", "right", "normal"])])
            pidx = 0
        else:
            aloc = self.participant_conditions.iloc[self.pointer]["auditory_localization"]
            vloc = self.participant_conditions.iloc[self.pointer]["visualcue_localization"]
            pidx = int(self.participant_conditions.iloc[self.pointer]["participant_idx"])
        self.pointer += 1

        if (aloc == "left" and vloc == "left") or (aloc == "right" and vloc == "right"):
            cc = "congruent"
        elif vloc == "normal":
            cc = "neutral"
        else:
            cc = "incongruent"

        info.update({"frame_annotations": {"condition_congruency": cc,
                                           "auditory_localization": aloc,
                                           "visualcue_localization": vloc,
                                           "participant_idx": pidx}})
        properties = {**properties,
                      "dummy_prop": (dummy_prop, "toggle", (True, False))}

        self.movie_player.exchange_object(**info["frame_annotations"])
        print("Running HRI video player")
        return info, properties


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--client", default="robot", type=str, required=False, choices=["robot", "human", "viewer"],
                        help="The playback mode depending on the display screen. "
                             "The 'human' mode captures keystrokes in addition to 'robot' mode playback, "
                             "whereas 'viewer' displays the scores on the big screen")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_dir = "datasets/wtmsimgaze2020/videos"

    if args.client == "human_DISABLED_FUNCTIONALITY":
        width, height = 1600, 800
        play_audio = True

        # define the reader
        reader = "WTMSimGazeSampleReader"
        sampler = "WTMSimGazeSample"
        # sampler_properties = {"show_saliency_map": True, "enable_transform_overlays":False, "color_map": "bone"}
        sampler_properties = {"show_saliency_map": True, "participant": None}

        SampleRegistrar.scan()
        ReaderRegistrar.scan()

        plotter = OpenCV()
        video_source = ReaderRegistrar.registry[reader](mode="d")
        video = SampleRegistrar.registry[sampler](video_source, w_size=1, width=width, height=height,
                                                  enable_audio=play_audio,
                                                  audio_reader=("AudioCapture", {"channels": 1}))
        movie_player = HRIMoviePlayer()
        movie_player.activate_communication("exchange_object", mode="listen")
        while True:
            obj, = movie_player.exchange_object(None, None, None)
            if obj is not None:
                movie_player.play_video_capture(plotter, video, n_frames=2000, enable_audio=play_audio, video_properties=sampler_properties)
                video.goto(0)

    elif args.client == "human":
        fullscreen = False
        metrics_log = pd.DataFrame(columns=["participant_idx", "conditions_congruency",
                                            "auditory_localization", "visualcue_localization",
                                            "pred_direction", "pred_accuracy"])

        "participant_idx': 6, 'conditions_congruency': 'neutral', 'auditory_localization': 'right', 'visualcue_localization': 'normal'"
        # the video playing client would run this script

        movie_player = HRIMoviePlayer()
        movie_player.activate_communication("exchange_object", mode="listen")
        while True:
            obj, = movie_player.exchange_object(None, None, None)
            if obj is not None:
                if obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "left":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "con_gaze_left_sound_left.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f','j']),
                     ])
                elif obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "right":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "inc_gaze_right_sound_left.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "left":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "inc_gaze_left_sound_right.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "right":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "con_gaze_right_sound_right.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])
                elif obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "normal":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "neutral_gaze_center_sound_left.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "normal":
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)(os.path.join(video_dir, "neutral_gaze_center_sound_right.mp4"),
                                                          fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])

                else:
                    returns = Parallel(n_jobs=1, prefer="threads")(
                        [delayed(movie_player.play_video)( os.path.join(video_dir, "neutral_gaze_center_sound_right.mp4"),
                            fullscreen=fullscreen),
                         delayed(movie_player.capture_key)(['f', 'j']),
                         ])
                predictions = {}
                if returns[1] == "f":
                    predictions.update(pred_direction="left")
                elif returns[1] == "j":
                    predictions.update(pred_direction="right")

                if (predictions["pred_direction"] == "left" and obj["auditory_localization"] == "left") or \
                        (predictions["pred_direction"] == "right" and obj["auditory_localization"] == "right"):
                    predictions.update(pred_accuracy=1.0)
                else:
                    predictions.update(pred_accuracy=0.0)

                obj.update(**predictions)
                metrics_log = metrics_log.append(obj, ignore_index=True)
                metrics_log.to_csv("logs/app_metrics/hriavcongruence_human.csv")
                print(metrics_log)
                print(obj)

    elif args.client == "robot" or  args.client == "human":
        fullscreen = True if args.client == "robot" else False
        # the video playing client would run this script

        movie_player = HRIMoviePlayer()
        movie_player.activate_communication("exchange_object", mode="listen")
        while True:
            obj, = movie_player.exchange_object(None, None, None)
            if obj is not None:
                if obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "left":
                    movie_player.play_video(os.path.join(video_dir, "con_gaze_left_sound_left.mp4"), fullscreen=fullscreen)
                elif obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "right":
                    movie_player.play_video(os.path.join(video_dir, "inc_gaze_right_sound_left.mp4"), fullscreen=fullscreen)
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "left":
                    movie_player.play_video(os.path.join(video_dir, "inc_gaze_left_sound_right.mp4"), fullscreen=fullscreen)
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "right":
                    movie_player.play_video(os.path.join(video_dir, "con_gaze_right_sound_right.mp4"), fullscreen=fullscreen)
                elif obj["auditory_localization"] == "left" and obj["visualcue_localization"] == "normal":
                    movie_player.play_video(os.path.join(video_dir, "neutral_gaze_center_sound_left.mp4"), fullscreen=fullscreen)
                elif obj["auditory_localization"] == "right" and obj["visualcue_localization"] == "normal":
                    movie_player.play_video(os.path.join(video_dir, "neutral_gaze_center_sound_right.mp4"), fullscreen=fullscreen)
                else: # default looks to the right if none of the above
                    movie_player.play_video(os.path.join(video_dir, "neutral_gaze_center_sound_right.mp4"), fullscreen=fullscreen)

                print(obj)

    elif args.client == "viewer":
        dash_app.vid_streamer = CV2VideoStreamer(0)
        #dash_app.stim_streamer = CV2VideoStreamer(os.path.join(video_dir, "neutral_gaze_center_sound_right.mp4"))
        wa = Thread(target=run_webapp)
        wa.start()
