import inspect
import argparse
import os
import json
import copy

from comet_ml import Experiment
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.callbacks import *

from gazenet.utils.helpers import flatten_dict
from gazenet.utils.registrar import *
from gazenet.bin.infer import infer


def store_config_log(config, logger, prefix="", filename="train_config.cfg"):
    config_dict = None

    if inspect.isclass(config):
        config_dict = {key: value for key, value in config.__dict__.items() if
                       not key.startswith('__') and not isinstance(value, classmethod) and not inspect.ismethod(
                           value)}
        config_dict.update({key: getattr(config, key)() for key, value in config.__dict__.items() if
                            not key.startswith('__') and (isinstance(value, classmethod) or inspect.ismethod(value))})
    elif isinstance(config, dict):
        config_dict = config

    config_dict = flatten_dict(config_dict, "", {})
    log_path = config.log_dir

    if config_dict is not None:
        if isinstance(logger, CometLogger):
            logger.experiment.log_parameters(config_dict, prefix=prefix)
            log_path = os.path.join(logger.save_dir, logger.experiment.project_name,
                                    config.comet_experiment_key)
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, filename), "w") as fp:
                json.dump(config_dict, fp)

        if isinstance(logger, TensorBoardLogger):
            logger.log_hyperparams(config_dict)
            log_path = logger.log_dir
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, filename), "w") as fp:
                json.dump(config_dict, fp)
    return log_path


def train(args, config, infer_configs=None):
    metrics = None
    log_path = config.log_dir
    experiment_name = config.experiment_name
    experiment_key = "_"

    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    if config.logger == "comet":
        logger = CometLogger(
            api_key=os.environ["COMET_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],  # Optional
            project_name=config.project_name,  # Optional
            save_dir=config.log_dir,
            experiment_name=config.experiment_name  # Optional
        )
        setattr(config, "comet_experiment_key", logger.experiment.id)
        experiment_key = logger.experiment.id
        log_path = store_config_log(config, logger=logger, prefix="train_config.")  # " + args.train_config + "."

    elif config.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=config.log_dir, name=config.project_name)
        setattr(config, "tensorboard_experiment_key", logger.log_dir)
        experiment_key = logger.log_dir.split("/")[-1]
        log_path = store_config_log(config, logger=logger, prefix="train_config.")
    else:
        logger = False

    # saves the checkpoint
    checkpoint_path = os.path.join(config.checkpoint_model_dir, experiment_name, experiment_key)
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=args.checkpoint_save_n_top,
        period=args.checkpoint_save_every_n_epoch,
        mode='min')

    # train
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, logger=logger)
    config.model_properties.update(train_dataset_properties=config.train_dataset_properties,
                                   val_dataset_properties=config.val_dataset_properties,
                                   test_dataset_properties=config.test_dataset_properties)

    if args.auto_lr_find:
        model_data = None
        model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                           val_store_image_samples=args.val_store_image_samples)
        trainer.tune(model)
    else:
        if hasattr(config, "model_data_name"):
            model_data = ModelDataRegistrar.registry[config.model_data_name](**config.model_properties)
            model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                               val_store_image_samples=args.val_store_image_samples,
                                                               **model_data.get_attributes())
            trainer.fit(model, model_data)
        else:
            model_data = None
            model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                               val_store_image_samples=args.val_store_image_samples)
            trainer.fit(model)

    last_checkpoint_file = os.path.join(checkpoint_path, "last_model.pt")
    torch.save(model.state_dict(), last_checkpoint_file)

    # run the models in inference
    for infer_config in infer_configs:
        if model_data is None:
            updated_infer_config = model.update_infer_config(log_path, last_checkpoint_file, config,
                                                             copy.deepcopy(infer_config), device=args.gpus)
        else:
            updated_infer_config = model_data.update_infer_config(log_path, last_checkpoint_file, config,
                                                             copy.deepcopy(infer_config), device=args.gpus)
        updated_infer_config.compute_metrics = args.compute_metrics
        inferer_metrics = infer(args, updated_infer_config)
        if metrics is None:
            metrics = inferer_metrics
        else:
            metrics = metrics.append(inferer_metrics, ignore_index=True)
    return metrics


def parse_args():
    trainer_summaries = "training configuration summaries:"
    for config_name in TrainingConfigRegistrar.registry.keys():
        config_summary = TrainingConfigRegistrar.registry[config_name].config_info()["summary"]
        config_example = TrainingConfigRegistrar.registry[config_name].config_info()["example"]
        trainer_summaries += ("\n  " + config_name + "\n     " + config_summary + "\n     example: " + config_example)

    inferer_summaries = "inference configuration summaries:"
    for config_name in InferenceConfigRegistrar.registry.keys():
        config_summary = InferenceConfigRegistrar.registry[config_name].config_info()["summary"]
        inferer_summaries += ("\n  " + config_name + "\n     " + config_summary)

    summaries = trainer_summaries + "\n\n\n\n" + inferer_summaries
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=summaries)

    parser.add_argument("--train_config", type=str, default="TrainPlayground001", required=False,
                        choices=TrainingConfigRegistrar.registry.keys(),
                        help="The training configuration. Select config from ../configs/train_config.py")

    parser.add_argument("--infer_configs", type=str, default=[], nargs='+', required=False,  # "InferMetricsGASPTrain"
                        choices=InferenceConfigRegistrar.registry.keys(),
                        help="The list of inference configurations. "
                             "This is needed for the metrics computation. Select config from ../configs/train_config.py")

    parser.add_argument("--train_config_file", type=str, required=False,
                        help="The json training configuration file (overrides train_config).")

    parser.add_argument("--infer_config_files", type=str, nargs='+', required=False,  # "InferMetricsGASPTrain"
                        help="The list of json inference configuration files (overrides infer_configs). "
                             "This is needed for the metrics computation.")

    parser.add_argument("--logger_name", type=str, required=False,
                        choices=["comet", "tensorboard", ""],
                        help="The logging framework name")

    parser.add_argument('--checkpoint_save_every_n_epoch', type=int, default=1000, help='Save model every n epochs')

    parser.add_argument('--checkpoint_save_n_top', type=int, default=3, help='Save top n model checkpoints')

    parser.add_argument('--val_store_image_samples', help='Store sampled validation images to logger',
                        action='store_true')

    parser.add_argument('--compute_metrics', help='Compute the metrics',
                        action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    TrainingConfigRegistrar.scan()
    InferenceConfigRegistrar.scan()
    args = parse_args()

    if args.train_config_file:
        with open(args.train_config_file) as fp:
            data = json.load(fp)
        config = TrainingConfigRegistrar.registry["TrainerBase"]
        config.__name__ = os.path.splitext(os.path.basename(args.train_config_file))[0]
        for data_key, data_val in data.items():
            setattr(config, data_key, data_val)
    else:
        config = TrainingConfigRegistrar.registry[args.train_config]

    # update config with args
    setattr(config, "compute_metrics", args.compute_metrics)
    if args.logger_name is not None:
        setattr(config, "logger", args.logger_name)

    # inference configs
    infer_configs = []
    for infer_config_name in args.infer_configs:
        infer_configs.append(InferenceConfigRegistrar.registry[infer_config_name])

    # scan the registrars
    InferenceRegistrar.scan()
    ReaderRegistrar.scan()
    SampleRegistrar.scan()
    FaceDetectorRegistrar.scan()
    AudioFeatureRegistrar.scan()

    ModelRegistrar.scan()
    ModelDataRegistrar.scan()

    # create metrics if enabled
    if args.compute_metrics:
        # scan the metrics registrar
        MetricsRegistrar.scan()

    # train
    metrics = train(args, config, infer_configs=infer_configs)
    if config.compute_metrics:
        print(metrics)


if __name__ == "__main__":
    main()
