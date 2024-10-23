"""
code from: https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py

Modified by Fares Abawi (fares.abawi@uni-hamburg.de)
"""

import random
import math

import numpy as np
import pandas as pd
import cv2

from gazenet.utils.registrar import *

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    return norm_s_map


def discretize_gt(gt):
    import warnings
    # warnings.warn('can improve the way GT is discretized')
    gt[gt > 0] = 255
    return gt / 255


@MetricsRegistrar.register
class SaliencyPredictionMetrics(object):
    def __init__(self, save_file="logs/metrics/salpred.csv", dataset_name="", video_name="",
                 metrics_list=["sim", "aucj", "aucs", "aucb", "nss", "cc", "kld", "ifg"],
                 map_key="frame_detections_gasp", metrics_mappings=None, *args, **kwargs):

        self.save_file = save_file
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        self.metrics_list = metrics_list
        self.map_key = map_key
        self.metrics_mappings = metrics_mappings

        self._dataset_name = dataset_name
        self._video_name = video_name

        if os.path.exists(save_file):
            self.scores = pd.read_csv(save_file, header=0)
        else:
            self.scores = pd.DataFrame(columns=["video_id", "dataset", "frames_len"] + metrics_list)
        self.accumulator = {metric: [] for metric in metrics_list}

        # NOTE: other_accumulator (gate_scores for example) cannot be loaded from a file, and will replace any previous runs
        self.other_scores = {}
        self.other_accumulator = {}

    def set_new_name(self, vid_name):
        collated_metrics = self.accumulate_metrics()
        self._video_name = vid_name
        self.accumulator = {metric: [] for metric in self.metrics_list}
        self.other_accumulator = {}
        return collated_metrics

    def save(self):
        self.scores.to_csv(self.save_file,  index=False)
        for o_score_name, o_scores in self.other_scores.items():
            o_scores.to_csv(self.save_file.replace(".csv", "_" + o_score_name + ".csv"), index=False)

    def accumulate_metrics(self, intermed_save=True):
        # accumulate metrics
        collated_metrics = {}
        collated_metrics["video_id"] = self._video_name
        collated_metrics["dataset"] = self._dataset_name
        collated_metrics["frames_len"] = 0

        for metric_name, metric_vals in self.accumulator.items():
            collated_metrics["frames_len"] = max(collated_metrics["frames_len"], len(metric_vals))
            if metric_vals:
                collated_metrics[metric_name] = np.nanmean(np.array(metric_vals))
        if collated_metrics:
            self.scores = self.scores.append(collated_metrics, ignore_index=True)

        # accumulate other scores
        other_scores = {o_score_name: {} for o_score_name in self.other_scores.keys()}
        for o_score_name in other_scores.keys():
            other_scores[o_score_name]["video_id"] = self._video_name
            other_scores[o_score_name]["dataset"] = self._dataset_name
            other_scores[o_score_name]["frames_len"] = 0

            other_scores[o_score_name]["frames_len"] = max(other_scores[o_score_name]["frames_len"],
                                                           len(self.other_accumulator[o_score_name]))
            if self.other_accumulator[o_score_name]:
                o_score_collated = np.nanmean(np.array(self.other_accumulator[o_score_name]), axis=0)
                for o_score_idx, o_score in enumerate(o_score_collated.tolist()):
                    if o_score:
                        other_scores[o_score_name][str(o_score_idx)] = o_score

                self.other_scores[o_score_name] = self.other_scores[o_score_name].append(other_scores[o_score_name], ignore_index=True)


        # save to file after every video
        if intermed_save:
            self.save()
        return collated_metrics, other_scores

    def add_metrics(self, returns, models, *args, **kwargs):
        metrics_args = {}
        eval_frame_id = 0
        baseline_imgs = []
        for idx_model, model_data in enumerate(models):
            for i, frame_dict in enumerate(returns[2 + idx_model][4]):
                # get the image frames
                for img_name in returns[2 + idx_model][1][i].keys():
                    if img_name in self.metrics_mappings.values():
                        img = returns[2 + idx_model][1][i][img_name]
                        metrics_args[list(self.metrics_mappings.keys())[list(self.metrics_mappings.values()).index(img_name)]] = img
                        eval_frame_id = frame_dict["frame_info"]["frame_id"]
                # get the scores info from the annotations if specified: scores should be a vector e.g. the gate scores
                if "scores_info" in self.metrics_mappings.keys():
                    for score_name in self.metrics_mappings["scores_info"]:
                        # TODO (fabawi): this should be agnostic to GASP, but right now it works only for GASP variants
                        try:
                            if score_name in frame_dict[self.map_key]:
                                if score_name in self.other_accumulator:
                                    self.other_accumulator[score_name].append(frame_dict[self.map_key][score_name][0][0])
                                else:
                                    column_list = [str(idx) for idx in range(frame_dict[self.map_key][score_name][0][0].shape[0])]
                                    if not score_name in self.other_scores:
                                        self.other_scores[score_name] = pd.DataFrame(columns=["video_id", "dataset", "frames_len"] + column_list)
                                    self.other_accumulator[score_name] = [frame_dict[self.map_key][score_name][0][0]]
                        except:
                            pass

        # extract all the frames besides the evaluation frame for creating the baseline map if needed
        if "gt_baseline" in self.metrics_mappings.keys():
            baseline_name = self.metrics_mappings["gt_baseline"]
            if "/" in baseline_name:
                metrics_args["gt_baseline"] = cv2.imread(baseline_name)
            else:
                info_list = returns[0]["info_list"]
                for i, info in enumerate(info_list):
                    if info["frame_info"]["frame_id"] != eval_frame_id:
                        baseline_img = returns[0]["grouped_video_frames_list"][i][baseline_name]
                        if baseline_img is not None:
                            baseline_imgs.append(baseline_img)
                if baseline_imgs:
                    baseline_imgs = np.nanmean(np.array(baseline_imgs), axis=0).astype(np.uint8)
                    metrics_args["gt_baseline"] = baseline_imgs
        metrics = self.compute_metrics(**metrics_args)
        if metrics is not None:
            for metric_name, metric_val in metrics.items():
                self.accumulator[metric_name].append(metric_val)
        return metrics

    def compute_metrics(self, pred_salmap=None, gt_fixmap=None, gt_salmap=None, gt_baseline=None):
        if pred_salmap is None or gt_fixmap is None or gt_salmap is None:
            return None
        else:
            try:
                pred_salmap = cv2.cvtColor(pred_salmap.copy(), cv2.COLOR_BGR2GRAY)
            except cv2.error:
                pass
            try:
                gt_fixmap = cv2.cvtColor(gt_fixmap.copy(), cv2.COLOR_BGR2GRAY)
            except cv2.error:
                pass
            try:
                gt_salmap = cv2.cvtColor(gt_salmap.copy(), cv2.COLOR_BGR2GRAY)
            except cv2.error:
                pass
            if gt_baseline is not None:
                try:
                    gt_baseline = cv2.cvtColor(gt_baseline.copy(), cv2.COLOR_BGR2GRAY)
                    gt_baseline = cv2.resize(gt_baseline, gt_salmap.shape)
                except cv2.error:
                    gt_baseline = None

            metrics = {}
            pred_salmap_minmax_norm = normalize_map(pred_salmap)

            if "aucj" in self.metrics_list:
                metrics["aucj"] = self.auc_judd(pred_salmap_minmax_norm, gt_fixmap)

            if "aucb" in self.metrics_list:
                metrics["aucb"] = self.auc_borji(pred_salmap_minmax_norm, gt_fixmap)

            if "aucs" in self.metrics_list:
                if gt_baseline is not None:
                    metrics["aucs"] = self.auc_shuff(pred_salmap_minmax_norm, gt_fixmap, gt_baseline)
                else:
                    metrics["aucs"] = np.nan

            if "nss" in self.metrics_list:
                metrics["nss"] = self.nss(pred_salmap, gt_fixmap)

            if "ifg" in self.metrics_list:
                if gt_baseline is not None:
                    metrics["ifg"] = self.infogain(pred_salmap_minmax_norm, gt_fixmap, gt_baseline)
                else:
                    metrics["ifg"] = np.nan

            # continous gts
            if "sim" in self.metrics_list:
                metrics["sim"] = self.similarity(pred_salmap, gt_salmap)

            if "cc" in self.metrics_list:
                metrics["cc"] = self.cc(pred_salmap, gt_salmap)

            if "kld" in self.metrics_list:
                metrics["kld"] = self.kldiv(pred_salmap, gt_salmap)

            return metrics

    @staticmethod
    def similarity(s_map, gt):
        # here gt is not discretized
        s_map = normalize_map(s_map)
        gt = normalize_map(gt)
        s_map = s_map / (np.sum(s_map) * 1.0)
        gt = gt / (np.sum(gt) * 1.0)
        x, y = np.where(gt > 0)
        sim = 0.0
        for i in zip(x, y):
            sim = sim + min(gt[i[0], i[1]], s_map[i[0], i[1]])
        return sim


    @staticmethod
    def auc_judd(s_map, gt):
        # ground truth is discrete, s_map is continous and normalized
        gt = discretize_gt(gt)
        # thresholds are calculated from the salience map, only at places where fixations are present
        thresholds = []
        for i in range(0, gt.shape[0]):
            for k in range(0, gt.shape[1]):
                if gt[i][k] > 0:
                    thresholds.append(s_map[i][k])

        num_fixations = np.sum(gt)
        # num fixations is no. of salience map values at gt >0

        thresholds = sorted(set(thresholds))

        # fp_list = []
        # tp_list = []
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            if np.max(gt) != 1.0:
                return np.nan
            if np.max(s_map) != 1.0:
                return np.nan

            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
            # this becomes nan when gt is full of fixations..this won't happen
            fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        return np.trapz(np.array(tp_list), np.array(fp_list))

    @staticmethod
    def auc_shuff(s_map, gt, other_map, n_splits=100, stepsize=0.1):
        # If there are no fixations to predict, return NaN
        if np.sum(gt) == 0:
            return np.nan

        # normalize saliency map
        # s_map = normalize_map(s_map)

        S = s_map.flatten()
        F = gt.flatten()
        Oth = other_map.flatten()

        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)

        # for each fixation, sample Nsplits values from the sal map at locations
        # specified by other_map

        ind = np.where(Oth > 0)[0]  # find fixation locations on other images

        Nfixations_oth = min(Nfixations, len(ind))
        randfix = np.full((Nfixations_oth, n_splits), np.nan)

        for i in range(n_splits):
            # randomize choice of fixation locations
            randind = np.random.permutation(ind.copy())
            # sal map values at random fixation locations of other random images
            randfix[:, i] = S[randind[:Nfixations_oth]]

        # calculate AUC per random split (set of random locations)
        auc = np.full(n_splits, np.nan)
        for s in range(n_splits):

            curfix = randfix[:, s]

            allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
            tp = np.zeros(len(allthreshes) + 2)
            fp = np.zeros(len(allthreshes) + 2)
            tp[-1] = 1
            fp[-1] = 1

            for i in range(len(allthreshes)):
                thresh = allthreshes[i]
                tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
                fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

            auc[s] = np.trapz(np.array(tp), np.array(fp))

        return np.mean(auc)

    @staticmethod
    def auc_borji(s_map, gt, splits=100, stepsize=0.1):
        gt = discretize_gt(gt)
        num_fixations = np.sum(gt).astype(np.int)

        num_pixels = s_map.shape[0] * s_map.shape[1]
        random_numbers = []
        for i in range(0, splits):
            temp_list = []
            for k in range(0, num_fixations):
                temp_list.append(np.random.randint(num_pixels))
            random_numbers.append(temp_list)

        aucs = []
        # for each split, calculate auc
        for i in random_numbers:
            r_sal_map = []
            for k in i:
                r_sal_map.append(s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]])
            # in these values, we need to find thresholds and calculate auc
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            r_sal_map = np.array(r_sal_map)

            # once threshs are got
            thresholds = sorted(set(thresholds))
            area = []
            area.append((0.0, 0.0))
            for thresh in thresholds:
                # in the salience map, keep only those pixels with values above threshold
                temp = np.zeros(s_map.shape)
                temp[s_map >= thresh] = 1.0
                num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
                tp = num_overlap / (num_fixations * 1.0)

                # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
                # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
                fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

                area.append((round(tp, 4), round(fp, 4)))

            area.append((1.0, 1.0))
            area.sort(key=lambda x: x[0])
            tp_list = [x[0] for x in area]
            fp_list = [x[1] for x in area]

            aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

        return np.mean(aucs)

    @staticmethod
    def nss(s_map, gt):
        gt = discretize_gt(gt)
        s_map_std_norm = (s_map - np.mean(s_map)) / np.std(s_map)

        x, y = np.where(gt == 1)
        temp = []
        for i in zip(x, y):
            temp.append(s_map_std_norm[i[0], i[1]])
        return np.mean(temp)

    @staticmethod
    def infogain(s_map, gt, baseline_map):
        gt = discretize_gt(gt)
        # assuming s_map and baseline_map are normalized
        eps = 2.2204e-16

        s_map = s_map / (np.sum(s_map) * 1.0)
        baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

        # for all places where gt=1, calculate info gain
        temp = []
        x, y = np.where(gt == 1)
        for i in zip(x, y):
            temp.append(np.log2(eps + s_map[i[0], i[1]]) - np.log2(eps + baseline_map[i[0], i[1]]))

        return np.mean(temp)

    @staticmethod
    def cc(s_map, gt):
        s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
        gt_norm = (gt - np.mean(gt)) / np.std(gt)
        a = s_map_norm
        b = gt_norm
        r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
        return r

    @staticmethod
    def kldiv(s_map, gt):
        s_map = s_map / (np.sum(s_map) * 1.0)
        gt = gt / (np.sum(gt) * 1.0)
        eps = 2.2204e-16
        return np.sum(gt * np.log(eps + gt / (s_map + eps)))




