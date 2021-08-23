import os

import numpy as np
from scipy.stats import gmean
from skimage.filters import window
import torch

from gazenet.utils.registrar import *
from gazenet.models.emotion_recognition.esr9.generator import pre_process_input_image
from gazenet.models.emotion_recognition.esr9.model import ESR
from gazenet.models.shared_components.gradcam.model import GradCAM
from gazenet.utils.sample_processors import InferenceSampleProcessor

MODEL_PATHS = {
    "esr9": os.path.join("gazenet", "models", "emotion_recognition", "esr9","checkpoints", "pretrained_esr9_orig"),
    "esr9_shared": "Net-Base-Shared_Representations.pt",
    "esr9_ensembles": "Net-Branch_{}.pt"}
TRG_CLASSES = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'}

INP_IMG_WIDTH = 96
INP_IMG_HEIGHT = 96
INP_IMG_MEAN = (0.0, 0.0, 0.0)
INP_IMG_STD = (1.0, 1.0, 1.0)
# INP_IMG_MEAN = [149.35457 / 255., 117.06477 / 255., 102.67609 / 255.]
# INP_IMG_STD = [69.18084 / 255., 61.907074 / 255., 60.435623 / 255.]
ENSEMBLES_NUM = 9


@InferenceRegistrar.register
class ESR9Inference(InferenceSampleProcessor):
    def __init__(self,  weights_path=MODEL_PATHS['esr9'],
                 shared_weights_basename=MODEL_PATHS["esr9_shared"],
                 ensembles_weights_baseformat=MODEL_PATHS["esr9_ensembles"],
                 enable_gradcam=True, ensembles_num=ENSEMBLES_NUM, trg_classes=TRG_CLASSES,
                 inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.short_name = "esr9"
        self._device = device

        self.ensembles_num = ensembles_num
        self.trg_classes = trg_classes
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        # load the model
        self.model = ESR(ensembles_num=ensembles_num)
        self.model.base.load_state_dict(torch.load(os.path.join(weights_path, shared_weights_basename), map_location=device))
        self.model.base.to(device)
        for en_idx, ensemble in enumerate(self.model.convolutional_branches, start=1):
            ensemble.load_state_dict(torch.load(os.path.join(weights_path, ensembles_weights_baseformat.format(en_idx)), map_location=device))
            ensemble.to(device)
        print("ESR-9 model loaded from", weights_path)
        self.model.to(device)
        self.model.eval()

        # load gradcam model
        if enable_gradcam:
            self.model_gradcam = GradCAM(self.model, device=device)
        else:
            self.model_gradcam = None


    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, faces_locations, source_frames_idxs, **kwargs):
        frames_idxs = range(len(video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for frame_id in frames_idxs:
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [],  # detected
                "emotions": [],  # detected
                "affects": [],  # detected
                "h_bboxes": []  # processed
            }}
            img = video_frames_list[frame_id]

            for id, face_local in enumerate(faces_locations[frame_id]):
                if not face_local:
                    continue
                sample_emotions = []
                sample_emotions_idx = []
                # sample_saliency = []
                # sample_affect = None

                (top, right, bottom, left) = face_local
                info["frame_detections_" + self.short_name]["h_bboxes"].append((left, top, right, bottom, id))

                # crop face image
                crop_img_face = img[top:bottom, left:right]
                crop_img_face = pre_process_input_image(crop_img_face, self.inp_img_width, self.inp_img_height,
                                                        self.inp_img_mean, self.inp_img_std)
                crop_img_face = crop_img_face.to(self._device, non_blocking=True)
                # emotion, affect, emotion_idx = _predict(crop_img_face)

                # computes ensemble prediction for affect
                emotion, affect = self.model(crop_img_face)

                # converts from Tensor to ndarray
                affect = np.array([a[0].cpu().detach().numpy() for a in affect])

                # normalizes arousal
                affect[:, 1] = np.clip((affect[:, 1] + 1) / 2.0, 0, 1)

                # computes mean arousal and valence as the ensemble prediction
                ensemble_affect = np.expand_dims(np.mean(affect, 0), axis=0)

                # concatenates the ensemble prediction to the list of affect predictions
                sample_affect = np.concatenate((affect, ensemble_affect), axis=0)
                info["frame_detections_" + self.short_name]["affects"].append((sample_affect, id))

                # converts from Tensor to ndarray
                emotion = np.array([e[0].cpu().detach().numpy() for e in emotion])

                # gets number of classes
                num_classes = emotion.shape[1]

                # computes votes and add label to the list of emotions
                emotion_votes = np.zeros(num_classes)
                for e in emotion:
                    e_idx = np.argmax(e)
                    sample_emotions_idx.append(e_idx)
                    sample_emotions.append(self.trg_classes[e_idx])
                    emotion_votes[e_idx] += 1

                # concatenates the ensemble prediction to the list of emotion predictions
                sample_emotions.append(self.trg_classes[np.argmax(emotion_votes)])
                info["frame_detections_" + self.short_name]["emotions"].append((sample_emotions, id))

                if self.model_gradcam is not None:
                    sample_saliency = self.model_gradcam.grad_cam(crop_img_face, sample_emotions_idx)
                    sample_saliency = np.array([s.cpu().detach().numpy() for s in sample_saliency])
                    info["frame_detections_" + self.short_name]["saliency_maps"].append((sample_saliency, id))

            info_list[frame_id].update(**info)
        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, video_frames_list=None, faces_locations=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        pad = features["preproc_pad_len"]
        lim = features["preproc_lim_len"]
        if video_frames_list is not None:
            video_frames_list = list(video_frames_list)
            features["video_frames_list"] = video_frames_list[:lim] + [video_frames_list[lim]] * pad
        if faces_locations is not None:
            faces_locations = list(faces_locations)
            features["faces_locations"] = faces_locations[:lim] + [faces_locations[lim]] * pad
        return features

    def annotate_frame(self, input_data, plotter,
                       show_det_saliency_map=True,
                       show_det_emotion_label=False,
                       show_det_valence_arousal_label=False,
                       show_det_head_bbox=True,
                       ensemble_aggregation="sum",
                       hanning_face=True,
                       keep_unmatching_ensembles=False,
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        properties = {**properties, "show_det_saliency_map": (show_det_saliency_map, "toggle", (True, False)),
                      "show_det_emotion_label": (show_det_emotion_label, "toggle", (True, False)),
                      "show_det_valence_arousal_label": (show_det_valence_arousal_label, "toggle", (True, False)),
                      "show_det_head_bbox": (show_det_head_bbox, "toggle", (True, False))}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": grouped_video_frames["PLOT"] + [["det_source_" + self.short_name,
                                                                         "det_transformed_" + self.short_name]],
                                "det_source_" + self.short_name: grouped_video_frames["captured"],
                                "det_transformed_" + self.short_name: grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}

        if grabbed_video:
            collated_saliency_maps = np.zeros_like(grouped_video_frames["captured"])
            if show_det_head_bbox or show_det_saliency_map or show_det_emotion_label:
                for h_bbox, saliency_map, emotion, affect in zip(info["frame_detections_" + self.short_name]["h_bboxes"],
                                                                 info["frame_detections_" + self.short_name]["saliency_maps"],
                                                                 info["frame_detections_" + self.short_name]["emotions"],
                                                                 info["frame_detections_" + self.short_name]["affects"],):
                    xmin_h_bbox, ymin_h_bbox, xmax_h_bbox, ymax_h_bbox, participant_id = h_bbox
                    if show_det_head_bbox:
                        frame_source = plotter.plot_bbox(grouped_video_frames["det_source_" + self.short_name], (xmin_h_bbox, ymin_h_bbox),
                                                         (xmax_h_bbox, ymax_h_bbox), color_id=participant_id)
                        grouped_video_frames["det_source_" + self.short_name] = frame_source
                    if show_det_saliency_map:
                        if keep_unmatching_ensembles:
                            filtered_saliency_map = saliency_map[0]
                        else:
                            # remove gradcams from wrong classifications
                            filter_idxs = [flt_idx for flt_idx, flt_cls in enumerate(emotion[0]) if
                                           flt_cls == emotion[0][-1]]
                            filtered_saliency_map = np.take(saliency_map[0], filter_idxs[:-1], axis=0)

                        # aggregate the ensembles into a single saliency plot
                        if ensemble_aggregation == "amean":
                            face_saliency = np.nanmean(filtered_saliency_map, axis=0)
                        elif ensemble_aggregation == "gmean":
                            face_saliency = gmean(filtered_saliency_map, axis=0)
                        elif ensemble_aggregation == "sum":
                            face_saliency = np.sum(filtered_saliency_map, axis=0)
                        else:
                            face_saliency = filtered_saliency_map[0]
                        # project saliency to full image
                        if hanning_face:
                            face_saliency = face_saliency * window("hann", face_saliency.shape)
                        face_saliency = plotter.plot_color_map(np.uint8(255 * face_saliency), color_map=color_map)
                        collated_saliency_maps += plotter.plot_alpha_overlay(collated_saliency_maps, face_saliency,
                                                                             xy_min=(xmin_h_bbox, ymin_h_bbox),
                                                                             xy_max=(xmax_h_bbox, ymax_h_bbox), alpha=1)
                    if show_det_emotion_label:
                        grouped_video_frames["det_transformed_" + self.short_name] = \
                            plotter.plot_text(grouped_video_frames["det_transformed_" + self.short_name],
                                              emotion[0][-1],
                                              (xmin_h_bbox, ymax_h_bbox + 10),
                                              color_id=emotion[1])
                    if show_det_valence_arousal_label:
                        grouped_video_frames["det_transformed_" + self.short_name] = \
                            plotter.plot_text(grouped_video_frames["det_transformed_" + self.short_name],
                                              "valence: " + str(np.round(affect[0][-1][0],3)),
                                              (xmin_h_bbox, ymax_h_bbox + 30),
                                              color_id=emotion[1])
                        grouped_video_frames["det_transformed_" + self.short_name] = \
                            plotter.plot_text(grouped_video_frames["det_transformed_" + self.short_name],
                                              "arousal: " + str(np.round(affect[0][-1][1], 3)),
                                              (xmin_h_bbox, ymax_h_bbox + 50),
                                              color_id=emotion[1])
                if show_det_saliency_map:
                    frame_transformed = \
                        plotter.plot_alpha_overlay(grouped_video_frames["det_transformed_" + self.short_name],
                                                   collated_saliency_maps,
                                                   alpha=0.4 if enable_transform_overlays else 1.0)
                    grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties
