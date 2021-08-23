import argparse
import subprocess
import inspect
import json
from shutil import copyfile, rmtree
import os

from PIL import Image
import scipy.io as sio
import numpy as np

from gazenet.utils.registrar import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--working_dir", default="./", type=str,
                        help="The working directory in which the directory structure is "
                             "built and downloaded files are stored.")

    parser.add_argument("--scripts", type=str, nargs='+', required=True,
                        help="The list of script names to be executed")

    return parser.parse_args()


def postprocess_get_from_stavis(dst_dir):
    #  after preprocessing, read preprocessed groundtruth from stavis and copy
    #   content to the preprocessed directory.

    stavis_img_dir = os.path.join(dst_dir, "datasets", "stavis_preprocessed")
    data_dir = os.path.join(dst_dir, "datasets", "processed", "Grouped_frames")

    for root, subdirs, files in os.walk(data_dir):
        print("scanning", root)
        for subdir in subdirs:
            tgt_dir = root.replace(data_dir, "")[1:]
            stavis_img_file = os.path.join(stavis_img_dir, tgt_dir, "maps", "eyeMap_" + f"{subdir:0>5}" + ".jpg")
            if os.path.isfile(stavis_img_file):
                copyfile(stavis_img_file, os.path.join(root, subdir, "transformed_salmap_1.jpg"))
            stavis_mat_file = os.path.join(stavis_img_dir, tgt_dir, "fixMap_" + f"{subdir:0>5}" + ".mat")
            if os.path.isfile(stavis_mat_file):
                tmp_mat = sio.loadmat(stavis_mat_file)
                binmap_np = np.array(
                    Image.fromarray(tmp_mat['eyeMap'].astype(float)).resize((120, 120), resample=Image.BILINEAR)) > 0
                fixmap = Image.fromarray((255 * binmap_np).astype('uint8'))
                fixmap.save(os.path.join(root, subdir, "transformed_fixmap_1.jpg"))


def generate_config_files(dst_dir):
    # generate the json files from the classes in infer_config.py and train_config.py

    # training configurations
    TrainingConfigRegistrar.scan()

    for config_name, config in TrainingConfigRegistrar.registry.items():
        if inspect.isclass(config):
            config_dict = {key: value for key, value in zip(dir(config), [getattr(config, k) for k in dir(config)]) if
                           not key.startswith('__') and not isinstance(value, classmethod) and not inspect.ismethod(
                               value)}
            config_dict.update(
                {key: getattr(config, key)() for key, value in
                 zip(dir(config), [getattr(config, k) for k in dir(config)])
                 if
                 not key.startswith('__') and (isinstance(value, classmethod) or inspect.ismethod(value))})
        elif isinstance(config, dict):
            config_dict = config

        with open(os.path.join(dst_dir, "gazenet", "configs", "train_configs", config_name + ".json"), 'w') as fp:
            json.dump(config_dict, fp, indent=4)

    print("Generated train configs to %", os.path.join(dst_dir, "gazenet", "configs", "train_configs"))

    # inference configurations
    InferenceConfigRegistrar.scan()

    for config_name, config in InferenceConfigRegistrar.registry.items():
        if inspect.isclass(config):
            config_dict = {key: value for key, value in zip(dir(config), [getattr(config, k) for k in dir(config)]) if
                           not key.startswith('__') and not isinstance(value, classmethod) and not inspect.ismethod(
                               value)}
            config_dict.update({key: getattr(config, key)() for key, value in
                                zip(dir(config), [getattr(config, k) for k in dir(config)]) if
                                not key.startswith('__') and (
                                            isinstance(value, classmethod) or inspect.ismethod(value))})
        elif isinstance(config, dict):
            config_dict = config

        with open(os.path.join(dst_dir, "gazenet", "configs", "infer_configs", config_name + ".json"), 'w') as fp:
            json.dump(config_dict, fp, indent=4)

    print("Generated infer configs to %", os.path.join(dst_dir, "gazenet", "configs", "infer_configs"))

def clean_temp(dst_dir):
    rmtree(os.path.join(dst_dir, "temp"))
    os.mkdir(os.path.join(dst_dir, "temp"))


def main():
    args = parse_args()

    for script in args.scripts:
        if script == "postprocess_get_from_stavis":
            postprocess_get_from_stavis(args.working_dir)
        if script == "generate_config_files":
            generate_config_files(args.working_dir)
        if script == "clean_temp":
            clean_temp(args.working_dir)


if __name__ == "__main__":
    main()
