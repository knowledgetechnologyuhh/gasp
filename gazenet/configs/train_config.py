from gazenet.utils.registrar import *


@TrainingConfigRegistrar.register
class TrainerBase(object):
    inferer_name = ""
    model_name = ""
    model_properties = {}
    log_dir = ""
    logger = ""  # comet, tensorboard
    project_name = ""
    experiment_name = model_name
    checkpoint_model_dir = ""
    train_dataset_properties = {}
    val_dataset_properties = {}
    test_dataset_properties = {}

    @classmethod
    def config_info(cls):
        return {"summary": "This is base class and cannot be used directly.",
                "example": "None"}

@TrainingConfigRegistrar.register
class GASPExp001_1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPEncConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}
    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: 1x1 convolutional variant. Not included in the paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_ALSTM1x1Conv(object):
    
    inferer_name = "GASPInference"
    model_name = "GASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: ALSTM variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_Add1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPEncAddConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: Additive variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_SE1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPSEEncConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: SE variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}



@TrainingConfigRegistrar.register
class GASPExp001_GMU1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPEncGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: GMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_GMUALSTM1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: AGMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_ALSTMGMU1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: LAGMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_DAMALSTMGMU1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: DAM + LAGMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_DAMGMU1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPDAMEncGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: DAM + GMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp001_DAMGMUALSTM1x1Conv(object):

    inferer_name = "GASPInference"
    model_name = "GASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Static GASP: DAM + AGMU variant.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: Sequential ALSTM variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_4(GASPExp002_SeqALSTM1x1Conv_2):

    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_6(GASPExp002_SeqALSTM1x1Conv_2):

    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_8(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_10(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_12(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_14(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_16(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_2Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_4Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_6Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_8Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_10Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_12Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_14Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTM1x1Conv_16Norm(GASPExp002_SeqALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: LARGMU variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_4(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_6(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_8(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_10(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_12(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_14(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_16(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_2Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_4Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_6Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_8Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_10Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_12Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_14Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqALSTMGMU1x1Conv_16Norm(GASPExp002_SeqALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: ARGMU variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_4(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_6(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": False}



@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_8(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_10(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_12(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_14(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_16(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_2Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_4Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_6Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_8Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_10Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_12Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_14Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqGMUALSTM1x1Conv_16Norm(GASPExp002_SeqGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_4(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_6(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_8(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):
    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_10(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_12(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_14(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_16(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_2Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_4Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_6Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_8Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_10Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_12Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_14Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMGMUALSTM1x1Conv_16Norm(GASPExp002_SeqDAMGMUALSTM1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_4(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_6(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_8(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_10(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_12(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_14(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_16(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": False}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_2Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 2, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_4Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 4, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_6Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 6, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_8Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_10Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_12Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 12, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_14Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 14, "sequence_norm": True}

@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMALSTMGMU1x1Conv_16Norm(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 16, "sequence_norm": True}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 2}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: RGMU variant. Exp_Variant_[N] ->"
                           "N: Context size;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_4(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 4}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_6(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 6}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_8(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 8}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_10(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 10}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_12(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 12}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_14(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 14}


@TrainingConfigRegistrar.register
class GASPExp002_SeqRGMU1x1Conv_16(GASPExp002_SeqRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 16}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_2(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 2}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + RGMU variant. Exp_Variant_[N]{Norm} ->"
                           "N: Context size; Norm: Enable temporal normalization;",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_4(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 4}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_6(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 6}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_8(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 8}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_10(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 10}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_12(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 12}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_14(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 14}


@TrainingConfigRegistrar.register
class GASPExp002_SeqDAMRGMU1x1Conv_16(GASPExp002_SeqDAMRGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncRGMUConv"
    model_properties = {"modalities": 5, "in_channels": 3, "batch_size": 4, "sequence_len": 16}


# different Stage-1 SP
@TrainingConfigRegistrar.register
class GASPExp003_SeqDAMALSTMGMU1x1Conv_10Norm_TASED(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_tased", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant inferring on TASED. ",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp003_SeqDAMALSTMGMU1x1Conv_10Norm_UNISAL(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_unisal", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant inferring on UNISAL.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp003_SeqDAMALSTMGMU1x1Conv_10Norm_STAViS_1(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    train_dataset_properties = {"csv_file": "datasets/processed/train_stavis_1.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/test_stavis_1.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_stavis_1.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant running on STAViS (1st fold).",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp003_SeqDAMALSTMGMU1x1Conv_10Norm_STAViS_2(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    train_dataset_properties = {"csv_file": "datasets/processed/train_stavis_2.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/test_stavis_2.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_stavis_2.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}


    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant running on STAViS (2nd fold).",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp003_SeqDAMALSTMGMU1x1Conv_10Norm_STAViS_3(GASPExp002_SeqDAMALSTMGMU1x1Conv_2):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 5, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    train_dataset_properties = {"csv_file": "datasets/processed/train_stavis_3.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                         "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/test_stavis_3.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_stavis_3.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_stavis", "det_transformed_esr9",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}


    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant running on STAViS (3rd fold).",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


# ablation experiments {GE}{GF}{FER}
@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_000(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 2, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_001(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}

@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_010(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_011(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_100(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_101(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_8Norm_110(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 8, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", 
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 8) variant. "
                           "Exp_variant_{GE: True}{GF: True}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_000(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 2, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_001(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}

@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_010(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_011(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_100(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_101(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: True}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMGMUALSTM1x1Conv_10Norm_110(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncGMUALSTMConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", 
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + ARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: True}{FER: False}. Not in paper.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_000(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 2, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: False}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_001(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: False}{FER: True}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}

@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_010(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: False}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_011(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_vidgaze"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_vidgaze"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: False}{GF: True}{FER: True}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_100(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 3, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: False}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_101(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                       "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                                        "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: False}{FER: True}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class GASPExp004_SeqDAMALSTMGMU1x1Conv_10Norm_110(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPDAMEncALSTMGMUConv"
    model_properties = {"modalities": 4, "batch_size": 4, "sequence_len": 10, "sequence_norm": True}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "gasp_runs"
    experiment_name = model_name

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv",
                                "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", 
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv",
                              "video_dir": "datasets/processed/Grouped_frames",
                              "inp_img_names_list": ["captured", "det_transformed_dave",
                                                       "det_transformed_vidgaze", "det_transformed_gaze360"],
                              "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv",
                               "video_dir": "datasets/processed/Grouped_frames",
                               "inp_img_names_list": ["captured", "det_transformed_dave",
                                                        "det_transformed_vidgaze", "det_transformed_gaze360"],
                               "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "Sequential GASP: DAM + LARGMU (Context Size = 10) variant. "
                           "Exp_variant_{GE: True}{GF: True}{FER: False}.",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 100 --max_epochs 10000 "
                           "--checkpoint_save_every_n_epoch 100 --checkpoint_save_n_top 3"}


@TrainingConfigRegistrar.register
class TrainPlayground001(object):

    inferer_name = "GASPInference"
    model_name = "SequenceGASPEncALSTMGMUConv"
    model_properties = {"modalities": 5, "sequence_len": 5, "sequence_norm": False}

    log_dir = "logs"
    logger = "comet"  # comet, tensorboard
    project_name = "testing_gasp"
    experiment_name = model_name + "_TRAINONLY_BADVAL"

    checkpoint_model_dir = os.path.join("gazenet", "models", "saliency_prediction",
                                        "gasp", "checkpoints", "pretrained_" +
                                        str.lower(experiment_name))

    train_dataset_properties = {"csv_file": "datasets/processed/train_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured","det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    val_dataset_properties = {"csv_file": "datasets/processed/validation_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    test_dataset_properties = {"csv_file": "datasets/processed/test_ave.csv", "video_dir": "datasets/processed/Grouped_frames",
                                "inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9", "det_transformed_vidgaze", "det_transformed_gaze360"],
                                "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    @classmethod
    def config_info(cls):
        return {"summary": "This is a playground configuration which is unstable "
                           "but can be used for quickly testing and visualizing models",
                "example": "python3 gazenet/bin/train.py --train_config " + cls.__name__ +
                           " --gpus \"0\" --check_val_every_n_epoch 500 --max_epochs 5000 "
                           "--checkpoint_save_every_n_epoch 1000 --checkpoint_save_n_top 3"}
