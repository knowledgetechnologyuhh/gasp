import argparse
import errno
import os
import glob
import shutil
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--working_dir", default="./", type=str,
                        help="The working directory in which the directory structure is "
                             "built and downloaded files are stored.")

    parser.add_argument("--datasets", type=str, nargs='+', required=False,
                        help="The list of dataset names to be downloaded")

    parser.add_argument("--models", type=str, nargs='+', required=False,
                        help="The list of model names to be downloaded")

    return parser.parse_args()


def copy_dir(src_dir, dst_dir):
    try:
        shutil.copytree(src_dir, dst_dir, False, None)
    except (OSError, FileExistsError) as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src_dir, dst_dir)
        else:
            pass


def create_dir(parent_dir, dst_dir):
    dst_path = os.path.join(parent_dir, dst_dir)
    try:
        os.makedirs(dst_path, exist_ok=True)
        print("Directory '%s' creation succeeded" % dst_dir)
    except OSError as e:
        print("Directory '%s' creation failed")


def main():
    args = parse_args()

    # some boiler-plate dirs
    struct_dirs = ["temp", "logs", os.path.join("logs", "metrics")]
    for struct_dir in struct_dirs:
        create_dir(args.working_dir, struct_dir)

    # copy datasets directory if not already there (to restore structure from repo delete datasets/ in working_dir)
    if not os.path.isdir(os.path.join(args.working_dir, "datasets")):
        copy_dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "datasets"),
                   os.path.join(args.working_dir, "datasets"))

    # copy config directory if not already there (to restore structure from repo delete datasets/ in working_dir)
    if not os.path.isdir(os.path.join(args.working_dir, "gazenet", "configs", "infer_configs")):
        copy_dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "configs", "infer_configs"),
                 os.path.join(args.working_dir, "gazenet", "configs", "infer_configs"))
    if not os.path.isdir(os.path.join(args.working_dir, "gazenet", "configs", "train_configs")):
        copy_dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "configs", "train_configs"),
                 os.path.join(args.working_dir, "gazenet", "configs", "train_configs"))

    # download datasets
    if args.datasets:
        for dataset in args.datasets:
            dataset_script_path = os.path.join(args.working_dir, "datasets", dataset)
            bashCommand = "./download_dataset.sh"
            process = subprocess.Popen(bashCommand.split(),
                                       stdout=subprocess.PIPE, cwd=dataset_script_path, universal_newlines=True)
            for stdout_line in iter(process.stdout.readline, ""):
                print(stdout_line)
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, bashCommand.split())
            # output, error = process.communicate()

    # download models
    if args.models:
        for models in args.models:
            if "<...>" in models:
                base_models_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models")
                models = glob.glob(os.path.join(base_models_script_path,
                                                models.replace("<...>", "/**/download_model.sh")), recursive=True)
                models = map(os.path.dirname, models)
                models = [model.replace(base_models_script_path + os.sep, "") for model in models]
            else:
                models = [models]

            for model in models:
                base_model_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models", model)
                model_script_path = os.path.join(args.working_dir, "gazenet", "models", model)

                copy_dir(base_model_script_path, model_script_path)

                bashCommand = "./download_model.sh"
                process = subprocess.Popen(bashCommand.split(),
                                           stdout=subprocess.PIPE, cwd=model_script_path, universal_newlines=True)
                for stdout_line in iter(process.stdout.readline, ""):
                    print(stdout_line)
                process.stdout.close()
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, bashCommand.split())


if __name__ == "__main__":
    main()