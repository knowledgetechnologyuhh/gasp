#!/bin/bash/python3

import string
import random
import argparse
from collections import deque
import re
import os
import json
import copy
import threading
import sys
from typing import Any, Callable

from tqdm import tqdm
import torch
import numpy as np
import pandas
import cv2
from joblib import Parallel, delayed
import sounddevice as sd

from gazenet.utils.registrar import *
import gazenet.utils.sample_processors as sp
import gazenet.utils.capturers as cp
from gazenet.utils.dataset_processors import DataWriter, DataSplitter
from gazenet.utils.annotation_plotter import OpenCV
from gazenet.utils.helpers import stack_images, config_dict_to_class, replace_config_placeholder_args

LOG_INTERVAL = 8  # the saving interval (metrics[soon incl. videos, images, etc.] if configs.realtime_capture == True)
log_id_gen: Callable[[Any], str] = lambda: "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

pandas.set_option("display.max_columns", 15)
read_lock = threading.Lock()


def exec_video_extraction(video, face_detector, preprocessed_data, max_w_size=1, video_properties={}, plotter=None,
                          realtime_capture=False):
    if preprocessed_data is None:
        preprocessed_data = {}
        preprocessed_data["grabbed_video_list"] = deque(maxlen=max_w_size)
        preprocessed_data["grouped_video_frames_list"] = deque(maxlen=max_w_size)
        preprocessed_data["info_list"] = deque(maxlen=max_w_size)
        preprocessed_data["properties_list"] = deque(maxlen=max_w_size)
        preprocessed_data["video_frames_list"] = deque(maxlen=max_w_size)
        preprocessed_data["faces_locations"] = deque(maxlen=max_w_size)
        preprocessed_data["robot_controller_list"] = deque(maxlen=max_w_size)
    preprocessed_data = preprocessed_data.copy()
    extracted_data_list = video.extract_frames(extract_audio=False,
                                               realtime_indexing=True if realtime_capture else False)
    grabbed_video_list, grouped_video_frames_list, _, _, info_list, properties_list = \
        video.annotate_frames(extracted_data_list, plotter, **video_properties)
    with read_lock:
        preprocessed_data["grabbed_video_list"].extend(grabbed_video_list)
        preprocessed_data["grouped_video_frames_list"].extend(grouped_video_frames_list)
        preprocessed_data["info_list"].extend(info_list)
        preprocessed_data["properties_list"].extend(properties_list)

        video_frames_list = stack_images(grouped_video_frames_list, grabbed_video_list, plot_override=[["captured"]])
        preprocessed_data["video_frames_list"].extend(video_frames_list)
        if face_detector is not None:
            preprocessed_data["faces_locations"].extend(face_detector.detect_frames(video_frames_list))
    return preprocessed_data


def exec_audio_extraction(video, audio_feat_extractors, preprocessed_data, max_w_size=25, plotter=None):
    if preprocessed_data is None:
        preprocessed_data = {}
        preprocessed_data["grabbed_audio_list"] = deque(maxlen=max_w_size)
        preprocessed_data["audio_frames_list"] = deque(maxlen=max_w_size)
        preprocessed_data.update(
            **{audio_feat_name: deque(maxlen=max_w_size) for audio_feat_name in audio_feat_extractors.keys()})
        # preprocessed_data["audio_features"] = deque(maxlen=max_w_size)
        # preprocessed_data["hann_audio_frames"] = deque(maxlen=max_w_size)

    extracted_data_list = video.extract_frames(extract_video=False)
    _, _, grabbed_audio_list, audio_frames_list, _, _ = video.annotate_frames(extracted_data_list, plotter)
    if any(grabbed_audio_list):
        audio_idx = list(filter(lambda x: grabbed_audio_list[x], range(len(grabbed_audio_list))))
        with read_lock:
            try:
                audio_frames_list = audio_frames_list[audio_idx[0]]
                preprocessed_data["audio_frames_list"].extend(audio_frames_list)
                for audio_feat_name, audio_feat_extractor in audio_feat_extractors.items():
                    audio_feat = audio_feat_extractor.waveform_to_feature(audio_frames_list,
                                                                          rate=video.audio_cap.get(
                                                                              cp.AUCAP_PROP_SAMPLE_RATE))
                    preprocessed_data[audio_feat_name].extend(audio_feat)
            except:
                if video.audio_cap.get(cp.AUCAP_PROP_CHANNELS) == 1:
                    audio_frames_list = np.zeros((video.audio_cap.get(cp.AUCAP_PROP_BUFFER_SIZE),
                                                  video.audio_cap.get(cp.AUCAP_PROP_CHUNK_SIZE)),
                                                 dtype=np.float32)
                else:
                    audio_frames_list = np.zeros((video.audio_cap.get(cp.AUCAP_PROP_BUFFER_SIZE),
                                                  video.audio_cap.get(cp.AUCAP_PROP_CHUNK_SIZE),
                                                  video.audio_cap.get(cp.AUCAP_PROP_CHANNELS)),
                                                 dtype=np.float32)
                preprocessed_data["audio_frames_list"].extend(audio_frames_list)
                for audio_feat_name, audio_feat_extractor in audio_feat_extractors.items():
                    audio_feat = audio_feat_extractor.waveform_to_feature(audio_frames_list,
                                                                          rate=video.audio_cap.get(
                                                                              cp.AUCAP_PROP_SAMPLE_RATE))
                    preprocessed_data[audio_feat_name].extend(audio_feat)
            preprocessed_data["grabbed_audio_list"].extend(
                grabbed_audio_list * len(preprocessed_data["audio_frames_list"]))
    return preprocessed_data


def exec_inference(inferer, plotter, preprocessed_data, previous_data, source_frames_idxs=None,
                   inference_properties={}, preproc_properties={}, postproc_properties={}):
    with read_lock:
        postprocessed_data = inferer.preprocess_frames(**preprocessed_data, previous_data=previous_data,
                                                       **preproc_properties)
    extracted_data_list = inferer.extract_frames(**postprocessed_data, source_frames_idxs=source_frames_idxs)
    grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list = \
        inferer.annotate_frames(extracted_data_list, plotter, **inference_properties)
    grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list = \
        inferer.postprocess_frames(grabbed_video_list, grouped_video_frames_list,
                                   grabbed_audio_list, audio_frames_list, info_list, properties_list,
                                   **postproc_properties)
    return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list


def exec_application(executor, previous_data, application_properties={}):
    info_list, properties_list = executor.execute(previous_data=previous_data, **application_properties)
    return [], [], [], [], info_list, properties_list


def exec_robot_actuation(controller, returns, control_properties={}):
    if controller is not None:
        if "gaze_extraction_target" not in control_properties:
            raise KeyError("The gaze extraction target (name of the detector's key in the info list) must be specified")
        # this always assumes that the last model in the list contains the needed annotations
        for info in returns[-1][-1][4]:
            if "SCREEN_gaze_target" in info[control_properties["gaze_extraction_target"]]:
                fixation = info[control_properties["gaze_extraction_target"]]["SCREEN_gaze_target"]
                controller.gaze_at_screen((fixation[1], fixation[0]))
    return None


def infer(parsed_args, config, placeholder_args, *args, **kwargs):

    # replace config placeholders with $-prepended arguments
    config = replace_config_placeholder_args(config,
                                             config_type=InferenceConfigRegistrar,
                                             placeholder_args=placeholder_args)

    log_counter = 1  # increments on each sample iteration (needed only when realtime_capture == True)
    accumulated_metrics = None

    if not hasattr(config, "premodel_groups"):
        setattr(config, "premodel_groups", [])

    # create the robot controller if enabled
    if config.control_robot:
        controllers = []
        if hasattr(config, "robots"):
            for robot in config.robots:
                try:
                    ctrl = RobotControllerRegistrar.registry[robot](**config.controller_properties[robot])
                    ctrl.reset_gaze()
                    controllers.append(ctrl)
                except:
                    print(f"Warning: Could not connect to the robot controller: {robot}."
                          "Make sure you have installed all the requirements and dependencies/daemons are running")
        else:
            try:
                ctrl = RobotControllerRegistrar.registry[config.robot](**config.controller_properties)
                ctrl.reset_gaze()
                controllers.append(ctrl)
            except:
                print(f"Warning: Could not connect to the robot controller: {config.robot}."
                      "Make sure you have installed all the requirements and dependencies/daemons are running")

        if not controllers:
            controllers = None
    else:
        controllers = None

    # create the plotting helper
    plotter = OpenCV()

    # create the video object from camera
    if config.realtime_capture:
        config.sampler_properties.update(enable_audio=True, width=config.width, height=config.height)
        video = sp.SampleProcessor(**config.sampler_properties)
        video.load({"video_name": 0, "audio_name": 0, "has_audio": True})
    else:
        # or create the video object from a database
        config.reader_properties.update(mode="d")
        video_source = ReaderRegistrar.registry[config.reader](**config.reader_properties)
        config.sampler_properties.update(w_size=config.stride, enable_audio=config.enable_audio)
        video = SampleRegistrar.registry[config.sampler](video_source, **config.sampler_properties)
        # traverse dataset videos only
        if config.process_dataset_videos_only:
            config.datasplitter_properties.update(mode="r")
            dataset_splitter = DataSplitter(**config.datasplitter_properties)
            if video.short_name != "processed":
                dataset_samples = dataset_splitter.samples[(dataset_splitter.samples["scene_type"] == "Social") &
                                                           (dataset_splitter.samples["dataset"] == video.short_name)]
            else:
                dataset_samples = dataset_splitter.samples[(dataset_splitter.samples["scene_type"] == "Social")]
            dataset_iter = dataset_samples.iterrows()
            dataset = next(dataset_iter)
            if video.short_name != "processed":
                video.goto(dataset[1]["video_id"], by_index=False)
            else:
                video.goto(os.path.join(dataset[1]["dataset"], dataset[1]["video_id"]), by_index=False)
        elif config.datasplitter_properties.get("video_start_id", None) is not None:
            # video.goto("clip_50",by_index=False)  # choose the first video instead: next(video)
            video_goto_by_idx = config.datasplitter_properties.get("video_goto_by_idx", True)
            video_start_id = config.datasplitter_properties["video_start_id"]
            video.goto(int(video_start_id) if video_goto_by_idx else video_start_id, by_index=video_goto_by_idx)
        else:
            # video.goto("clip_50",by_index=False)  # choose the first video instead: next(video)
            video.goto(0)
    # get the fps
    w_fps = video.frames_per_sec()

    # create detectors
    audio_feature_extraction = {audio_feat_name: AudioFeatureRegistrar.registry[audio_feat](hop_len_sec=1 / (w_fps))
                                for audio_feat_name, audio_feat in config.audio_features.items()}
    face_detection = FaceDetectorRegistrar.registry[config.face_detector](device=config.device) \
        if config.face_detector is not None else None

    if config.write_images or config.write_annotations or config.write_videos:
        # create the data writer
        writer = DataWriter(video.short_name, video_name=video.reader.samples[video.index]["id"],
                            output_video_size=(video.reader.samples[video.index]["video_width"],
                                               video.reader.samples[video.index]["video_height"]),
                            frames_per_sec=w_fps,
                            # output_video_size=(1232, 504), frames_per_sec=25,
                            write_images=config.write_images,
                            write_annotations=config.write_annotations,
                            write_videos=config.write_videos)

        # create the nice bar so we can look at something while processing happens
        bar_writer = tqdm(desc="Write -> Video Nr: " + str(video.index), total=video.len_frames())

    if config.compute_metrics:
        # create the metrics
        metrics_logger = MetricsRegistrar.registry[config.metrics](save_file=config.metrics_save_file,
                                                                   dataset_name="realtime" if not hasattr(video, "short_name") else video.short_name,
                                                                   video_name=log_id_gen() if not hasattr(video,  "reader") else video.reader.samples[video.index]["id"],
                                                                   metrics_list=config.metrics_list,
                                                                   metrics_mappings=config.metrics_mappings)

        bar_metrics = tqdm(desc="Metrics -> Video Nr: " +
                                str(video.index if hasattr(video, "index") else video.rt_index),
                           total=video.len_frames())

    # create premodels (applications: flexible interface to internal or external execution pipelines)
    apps = [[ApplicationRegistrar.registry[app_data[0]](**app_data[3]) for app_data in app_group]
            for app_group in config.premodel_groups]

    inferers = []
    for model_group in config.model_groups:
        group_inferers = []
        for model_data in model_group:
            if model_data[0] in InferenceRegistrar.registry:
                group_inferers.append(InferenceRegistrar.registry[model_data[0]]
                                      (w_size=model_data[1], width=config.width, height=config.height,
                                       device=config.device, **model_data[3]))
            elif model_data[0] in ApplicationRegistrar.registry:
                group_inferers.append(ApplicationRegistrar.registry[model_data[0]]
                                      (**model_data[3]))
            else:
                raise NotImplementedError("Model name "+ str(model_data[0]) +" not found amongst inferers or applications")

        inferers.append(group_inferers)

    # create returns placeholder
    returns = [[]] * (len(config.premodel_groups) + len(config.model_groups))

    # initially run preprocessors
    preprocessed_vid_data_list = exec_video_extraction(video, face_detection, None, max_w_size=config.max_w_size,
                                                       video_properties=config.sampling_properties, plotter=plotter)

    preprocessed_aud_data_list = exec_audio_extraction(video, audio_feature_extraction, None, max_w_size=w_fps)
    preprocessed_data_list = {**preprocessed_vid_data_list, **preprocessed_aud_data_list}
    while True:
        try:
            for idx_model_group, model_group in enumerate(config.premodel_groups + config.model_groups):

                if idx_model_group < len(config.premodel_groups):
                    execution_recipe = []

                    for idx_app, app_data in enumerate(model_group):
                        execution_recipe.extend([delayed(exec_application)(apps[idx_model_group][idx_app],
                                                                           previous_data=returns[idx_model_group - 1]
                                                                           if idx_model_group != 0 else None,
                                                                           application_properties=app_data[4])])
                else:
                    if idx_model_group == len(config.premodel_groups):
                        # create execution recipe for extracting data
                        execution_recipe = [
                            delayed(exec_video_extraction)(video, face_detection, preprocessed_data_list,
                                                           video_properties=config.sampling_properties, plotter=plotter,
                                                           realtime_capture=config.realtime_capture),
                            delayed(exec_audio_extraction)(video, audio_feature_extraction, preprocessed_data_list),
                        ]
                    else:
                        execution_recipe = []
                    for idx_model, model_data in enumerate(model_group):
                        if isinstance(inferers[idx_model_group - len(config.premodel_groups)][idx_model],
                                      sp.InferenceSampleProcessor):
                            execution_recipe.extend([delayed(exec_inference)(
                                inferers[idx_model_group - len(config.premodel_groups)][idx_model], plotter,
                                preprocessed_data_list,
                                previous_data=returns[idx_model_group - 1]
                                if idx_model_group != 0 else None,
                                source_frames_idxs=model_data[2],
                                inference_properties=config.inference_properties,
                                preproc_properties=model_data[4],
                                postproc_properties=model_data[5])])
                        elif isinstance(inferers[idx_model_group - len(config.premodel_groups)][idx_model],
                                        sp.ApplicationProcessor):
                            execution_recipe.extend([delayed(exec_application)(
                                inferers[idx_model_group - len(config.premodel_groups)][idx_model],
                                previous_data=returns[idx_model_group - 1]
                                if idx_model_group != 0 else None,
                                application_properties=model_data[4])])

                returns[idx_model_group] = Parallel(n_jobs=config.n_jobs[idx_model_group], prefer="threads")(
                    execution_recipe)

                if idx_model_group == len(config.premodel_groups):
                    # update preprocessed data in the first model_group iteration
                    preprocessed_data_list = {**returns[idx_model_group][0], **returns[idx_model_group][1]}

                # write images and annotations
                if config.write_images or config.write_annotations or config.write_videos:
                    bar_writer.update(1)
                    if not any(returns[idx_model_group][idx_model_group]["grabbed_video_list"]):
                        raise IndexError
                    writer.add_detections(returns[idx_model_group], model_group)

                # compute metrics
                if config.compute_metrics:
                    bar_metrics.update(1)
                    # jump to next video when no more returns lefts
                    if not config.realtime_capture and \
                            not any(returns[idx_model_group][idx_model_group]["grabbed_video_list"]):
                        raise IndexError
                    frame_metrics = metrics_logger.add_metrics(returns[idx_model_group], model_group)
                    if config.realtime_capture and log_counter == 0:
                        metrics_logger.accumulate_metrics()
                        metrics_logger.set_new_name(log_id_gen())
                        log_counter += 1

                # customize this plotter as you like
                if config.visualize_images:
                    grabbed_video = False
                    try:
                        new_plot = {"PLOT": [[]]}
                        for idx_model, model_data in enumerate(model_group):
                            grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, \
                            info_list, properties_list = returns[idx_model_group][
                                2 + idx_model if idx_model_group == len(config.premodel_groups) else idx_model]
                            # if play_audio:
                            #     sd.play(np.array(audio_frames_list).flatten(), video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE))
                            #
                            if grouped_video_frames_list:
                                transformed_frames = stack_images(grouped_video_frames_list, grabbed_video_list)
                                for idx_frame, transformed_frame in enumerate(transformed_frames):
                                    if transformed_frame is not None:
                                        new_plot["transformed_" + model_data[0] + str(idx_frame)] = transformed_frame
                                        # new_plot["transformed_" + model_data[0] + str(idx_frame)] = transformed_frame
                                        new_plot["PLOT"][-1].extend(["transformed_" + model_data[0] + str(idx_frame)])
                                        grabbed_video = True
                                    else:
                                        pass
                                else:
                                    pass
                        if grabbed_video:
                            cv2.imshow("target_" + str(idx_model_group),
                                       stack_images(new_plot, grabbed_video_list=True))
                            # cv2.waitKey(int(1/w_fps * 100))
                            cv2.waitKey(1)
                    except (cv2.error, AttributeError, KeyError):
                        pass
                    # except (OSError):
                    #     pass


            if config.control_robot:
                for controller in controllers:
                    exec_robot_actuation(controller, returns, control_properties=config.control_properties)

            if config.realtime_capture:
                log_counter += 1
                log_counter %= LOG_INTERVAL

        # except (cv2.error, IndexError):
        except (cv2.error, IndexError, OSError):
            # videos progress text logger
            if hasattr(parsed_args, "progress_log_file") and parsed_args.progress_log_file:
                try:
                    with open(parsed_args.progress_log_file, 'a+') as prog_log:
                        prog_log.write("completed,"
                                       "videoid="+str(video.reader.samples[video.index]["id"])+","+
                                       "videoidx="+str(video.index))
                except:
                    print("Could not log video progress")

            # iterate videos for offline datasets
            if not config.realtime_capture:
                if config.process_dataset_videos_only:
                    try:
                        dataset = next(dataset_iter)
                        if video.short_name != "processed":
                            video.goto(dataset[1]["video_id"], by_index=False)
                        else:
                            video.goto(os.path.join(dataset[1]["dataset"], dataset[1]["video_id"]), by_index=False)
                    except StopIteration:
                        break
                else:
                    next(video)

            # get the fps
            w_fps = video.frames_per_sec()

            # start new images and annotations writer
            if config.write_images or config.write_annotations or config.write_videos:
                if config.write_annotations:
                    writer.dump_annotations()
                if config.write_videos:
                    writer.dump_videos()
                bar_writer.close()
                writer.set_new_name(video.reader.samples[video.index]["id"],
                                    output_vid_size=(video.reader.samples[video.index]["video_width"],
                                                     video.reader.samples[video.index]["video_height"]),
                                    fps=w_fps)
                # writer.set_new_name(video.reader.samples[video.index]["id"], output_vid_size=(1232, 504), fps=25)

                bar_writer = tqdm(desc="Write -> Video Nr: " + str(video.index), total=video.len_frames())

            if config.compute_metrics:
                bar_metrics.close()
                print(metrics_logger.set_new_name(video.reader.samples[video.index]["id"]))
                bar_metrics = tqdm(desc="Metrics -> Video Nr: " + str(video.index), total=video.len_frames())

            # create detectors
            audio_feature_extraction = {
                audio_feat_name: AudioFeatureRegistrar.registry[audio_feat](hop_len_sec=1 / (w_fps))
                for audio_feat_name, audio_feat in config.audio_features.items()}

            # re-run preprocessors
            for buff in preprocessed_vid_data_list.values():
                if isinstance(buff, deque):
                    buff.clear()
            for buff in preprocessed_aud_data_list.values():
                if isinstance(buff, deque):
                    buff.clear()
            preprocessed_vid_data_list = exec_video_extraction(video, face_detection, None,
                                                               max_w_size=config.max_w_size,
                                                               video_properties=config.sampling_properties,
                                                               realtime_capture=config.realtime_capture)

            preprocessed_aud_data_list = exec_audio_extraction(video, audio_feature_extraction, None, max_w_size=w_fps)

            preprocessed_data_list = {**preprocessed_vid_data_list, **preprocessed_aud_data_list}

    if config.compute_metrics:
        metrics_logger.accumulate_metrics()
        accumulated_metrics = metrics_logger.scores

    return accumulated_metrics


def parse_args():
    inferer_summaries = "configuration summaries:"
    for config_name in InferenceConfigRegistrar.registry.keys():
        config_summary = InferenceConfigRegistrar.registry[config_name].config_info()["summary"]
        inferer_summaries += ("\n  " + config_name + "\n     " + config_summary)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=inferer_summaries)

    parser.add_argument("--infer_config", type=str, default="InferPlayground001", required=False,
                        choices=InferenceConfigRegistrar.registry.keys(),
                        help="The inference configuration. Select config from ../configs/infer_config.py")
    parser.add_argument("--infer_config_file", type=str, required=False,
                        help="The json inference configuration file (overrides infer_config).")
    parser.add_argument("--gpu", type=int, required=False,
                        help="The GPU index with CUDA support for running the inferer.")
    parser.add_argument("--progress_log_file", type=str, required=False,
                        help="The text file showing completed video ids.")
    return parser.parse_known_args()


if __name__ == "__main__":
    InferenceConfigRegistrar.scan()
    args, unk_args = parse_args()

    if args.infer_config_file:
        with open(args.infer_config_file) as fp:
            config_dict = json.load(fp)
        config = config_dict_to_class(config_dict,
                                      config_type=InferenceConfigRegistrar,
                                      config_name=os.path.splitext(os.path.basename(args.infer_config_file))[0])
    else:
        config = InferenceConfigRegistrar.registry[args.infer_config]

    # update config with args
    if args.gpu is not None:
        setattr(config, "device", "cuda:" + str(args.gpu))

    # scan the registrars
    InferenceRegistrar.scan()
    ApplicationRegistrar.scan()
    ReaderRegistrar.scan()
    SampleRegistrar.scan()
    FaceDetectorRegistrar.scan()
    AudioFeatureRegistrar.scan()

    # create metrics if enabled
    if config.compute_metrics:
        # scan the metrics registrar
        MetricsRegistrar.scan()

    # create the robot controller if enabled
    if config.control_robot:
        # scan robot registrar
        RobotControllerRegistrar.scan()

    metrics = infer(args, config, unk_args)
    if config.compute_metrics:
        print(metrics)
