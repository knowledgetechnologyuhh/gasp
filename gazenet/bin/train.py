import inspect
import argparse
import os
import json
import copy
from collections import namedtuple

from comet_ml import Experiment
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from pytorch_lightning.callbacks import *

from gazenet.utils.helpers import flatten_dict, config_dict_to_class, replace_config_placeholder_args
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

        elif isinstance(logger, TensorBoardLogger):
            logger.log_hyperparams(config_dict)
            log_path = logger.log_dir
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, filename), "w") as fp:
                json.dump(config_dict, fp)

        elif isinstance(logger, WandbLogger):
            config_dict_with_prefix = {prefix + k: v for k, v in config_dict.items()}
            logger.experiment.config.update(config_dict_with_prefix)
            log_path = os.path.join(logger.save_dir, logger.experiment.project,
                                    config.wandb_experiment_key)
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, filename), "w") as fp:
                json.dump(config_dict, fp)

    return log_path


def train(parsed_args, config, placeholder_args, infer_configs=None, *args, **kwargs):
    # replace config placeholders with $-prepended arguments
    config = replace_config_placeholder_args(config,
                                             config_type=TrainingConfigRegistrar,
                                             placeholder_args=placeholder_args)

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
        log_path = store_config_log(config, logger=logger, prefix="train_config.")  # " + parsed_args.train_config + "."

        experiment_tags = getattr(config, "experiment_tags", [])
        logger.experiment.add_tags(experiment_tags)

    elif config.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=config.log_dir, name=config.project_name)
        setattr(config, "tensorboard_experiment_key", logger.log_dir)
        experiment_key = logger.log_dir.split("/")[-1]
        log_path = store_config_log(config, logger=logger, prefix="train_config.")

    elif config.logger == "wandb":
        experiment_tags = getattr(config, "experiment_tags", [])
        logger = WandbLogger(
            project=config.project_name,  # Optional
            save_dir=config.log_dir,
            name=config.experiment_name,  # Optional
            tags=experiment_tags  # Optional
        )
        setattr(config, "wandb_experiment_key", logger.experiment.id)
        experiment_key = logger.experiment.id
        log_path = store_config_log(config, logger=logger, prefix="train_config.")  # " + parsed_args.train_config + "."



    else:
        logger = False

    # saves the checkpoint
    checkpoint_path = os.path.join(config.checkpoint_model_dir, experiment_name, experiment_key)
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=parsed_args.checkpoint_monitor,
        dirpath=checkpoint_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=parsed_args.checkpoint_save_n_top,
        every_n_epochs=parsed_args.checkpoint_save_every_n_epoch,
        mode='min')

    # train
    trainer = pl.Trainer.from_argparse_args(parsed_args, checkpoint_callback=True, callbacks=[checkpoint_callback],
                                            logger=logger)
    config.model_properties.update(train_dataset_properties=config.train_dataset_properties,
                                   val_dataset_properties=config.val_dataset_properties,
                                   test_dataset_properties=config.test_dataset_properties)

    if parsed_args.auto_lr_find:
        model_data = None
        model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                           val_store_image_samples=parsed_args.val_store_image_samples)
        trainer.tune(model)
        return None
    else:
        # extract the metrics properties before running model if the inference pipeline is not to be accessed
        if config.compute_metrics and not infer_configs:
            # create the metrics
            try:
                metrics_save_file = config.metrics_save_file if config.metrics_save_file else os.path.join(log_path,
                                                                                                           "metrics.csv")
                metrics_logger = MetricsRegistrar.registry[config.metrics](save_file=metrics_save_file,
                                                                           dataset_name=None,
                                                                           video_name=None,
                                                                           metrics_mappings=config.metrics_mappings,
                                                                           metrics_list=config.metrics_list)
            except AttributeError:
                raise AttributeError("Cannot set --compute_metrics without an infer_config/s OR "
                                     "setting the metrics properties within the training configuration!")

        else:
            metrics_logger = None

        if hasattr(config, "model_data_name"):  # if model is split into module and data module
            model_data = ModelDataRegistrar.registry[config.model_data_name](**config.model_properties)
            model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                               val_store_image_samples=parsed_args.val_store_image_samples,
                                                               metrics_logger=metrics_logger,
                                                               **model_data.get_attributes())
            trainer.fit(model, model_data)
        else:  # if model is constructed as a module only
            model_data = None
            model = ModelRegistrar.registry[config.model_name](**config.model_properties,
                                                               val_store_image_samples=parsed_args.val_store_image_samples,
                                                               metrics_logger=metrics_logger)
            trainer.fit(model)

        last_checkpoint_file = os.path.join(checkpoint_path, "last_model.pt")
        torch.save(model.state_dict(), last_checkpoint_file)

        if metrics_logger is not None:
            trainer.test()  # trainer.test(ckpt_path="best")

        # run the models in inference
        for infer_config in infer_configs:
            if model_data is None:
                updated_infer_config = model.update_infer_config(log_path, last_checkpoint_file, config,
                                                                 copy.deepcopy(infer_config), device=parsed_args.gpus)
            else:
                updated_infer_config = model_data.update_infer_config(log_path, last_checkpoint_file, config,
                                                                      copy.deepcopy(infer_config),
                                                                      device=parsed_args.gpus)
            updated_infer_config.compute_metrics = parsed_args.compute_metrics
            inferer_metrics = infer(parsed_args, updated_infer_config, placeholder_args)
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

    parser.add_argument('--checkpoint_save_every_n_epoch', type=int, default=1000, help='Save model every n epochs.')

    parser.add_argument('--checkpoint_save_n_top', type=int, default=3, help='Save top n model checkpoints.')

    parser.add_argument('--checkpoint_monitor', type=str, default="val_loss",
                        help='Name of loss to monitor for checkpointing.')

    parser.add_argument('--val_store_image_samples', help='Store sampled validation images to logger.',
                        action='store_true')

    parser.add_argument('--compute_metrics', help='Compute the metrics.',
                        action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_known_args()


if __name__ == "__main__":
    TrainingConfigRegistrar.scan()
    InferenceConfigRegistrar.scan()
    args, unk_args = parse_args()

    if args.train_config_file:
        with open(args.train_config_file) as fp:
            config_dict = json.load(fp)
        config = config_dict_to_class(config_dict,
                                      config_type=TrainingConfigRegistrar,
                                      config_name=os.path.splitext(os.path.basename(args.train_config_file))[0])
    else:
        config = TrainingConfigRegistrar.registry[args.train_config]

    # update config with args
    setattr(config, "compute_metrics", args.compute_metrics)
    if args.logger_name is not None:
        setattr(config, "logger", args.logger_name)

    # inference configs
    infer_configs = []

    if args.infer_config_files:
        for infer_config_file in args.infer_config_files:
            with open(infer_config_file) as fp:
                config_dict = json.load(fp)
            infer_configs.append(config_dict_to_class(config_dict,
                                                      config_type=InferenceConfigRegistrar,
                                                      config_name=os.path.splitext(os.path.basename(infer_config_file))[
                                                          0]))
    else:
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
    metrics = train(args, config, unk_args, infer_configs=infer_configs)
    if config.compute_metrics:
        print(metrics)
