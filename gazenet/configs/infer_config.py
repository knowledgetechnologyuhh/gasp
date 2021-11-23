# TODO (fabawi): variables marked as AUTO should update automatically, but this won't happen. Place all vars in init

from gazenet.utils.registrar import *


@InferenceConfigRegistrar.register
class InferGeneratorAllModelsBase(object):
    # define the reader
    reader = ""
    sampler = ""

    reader_properties = {}
    sampler_properties = {}
    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                          "enable_transform_overlays": False, "color_map": "bone"}
    # sampler_properties = {"show_saliency_map": True}

    # define the face detector
    face_detector = "SFDFaceDetection"  # "MTCNNFaceDetection", "DlibFaceDetection"
    # define audio features needed by the models
    audio_features = {"audio_features": "MFCCAudioFeatures",
                      "hann_audio_frames": "WindowedAudioFeatures"}

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}
    #  model_name, window_size, source_frames_idxs, model_properties, preproc_properties, postproc_properties ->
    #  each in its own model_group: model groups are executed in order
    model_groups = [
        [["DAVEInference", 16, [15], {}, {},
          dict(postproc_properties, **{"plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_dave"]]})],
         ["ESR9Inference", 16, [15], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_source_esr9", "det_transformed_esr9"]]})],
         ["Gaze360Inference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_gaze360"]]})],
         ["VideoGazeInference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_vidgaze"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "jet"}

    # define the metrics calculator properties (only needed when compute_metrics=True)
    metrics = "SaliencyPredictionMetrics"
    metrics_list = ["aucj", "aucs", "cc", "nss", "sim"]
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/default.csv"

    # define the datasplitter properties (only needed when process_dataset_videos_only=True)
    datasplitter_properties = {"train_csv_file": "datasets/processed/train_ave.csv",
                               "val_csv_file": "datasets/processed/validation_ave.csv",
                               "test_csv_file": "datasets/processed/test_ave.csv"}

    # constants
    width, height = 500, 500  # the frame's width and height
    stride = 1  # the number of frames to capture per inference iteration. Should be lte than max_w_size
    max_w_size = 16  # AUTO: the largest window needed by any model
    enable_audio = True  # if only one of the models needs audio, then this should be set to True
    play_audio = False  # if any of the models employing audio has no source_frames_idxs. Check keep_audio in postproc_properties. DOES NOT WORK AT THE MOMENT
    realtime_capture = False  # capture audio/video in realtime (cam/mic)
    visualize_images = False  # visualize the plotters
    write_images = False  # if only realtime capture is False
    write_videos = False  # if only realtime capture is False
    write_annotations = False  # always set to False, since annotations not needed for training the models
    process_dataset_videos_only = True  # process videos only if they exist in the train,val,test sets if only realtime capture is False
    compute_metrics = False  # enable the metrics computation
    device = "cpu"  # the pytorch device to use for all models
    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    @classmethod
    def config_info(cls):
        return {"summary": "This generates the datasets images needed for a majority of the experiments. "
                           "Only dataset samples (Social) are generated. "
                           "The 'reader' and 'sampler' need to be set and does not write automatically. "}


@InferenceConfigRegistrar.register
class InferGeneratorAllModelsCoutrot1(InferGeneratorAllModelsBase):
    # define the reader
    reader = "Coutrot1SampleReader"
    sampler = "CoutrotSample"

    write_images = True  # if only realtime capture is False

    @classmethod
    def config_info(cls):
        return {"summary": "This generates the datasets images needed for a majority of the experiments. "
                           "Only dataset samples (Social) are generated. "
                           "It runs the 4 social cue modalities for Coutrot1. "}


@InferenceConfigRegistrar.register
class InferGeneratorAllModelsCoutrot2(InferGeneratorAllModelsBase):
    # define the reader
    reader = "Coutrot2SampleReader"
    sampler = "CoutrotSample"

    write_images = True  # if only realtime capture is False

    @classmethod
    def config_info(cls):
        return {"summary": "This generates the datasets images needed for a majority of the experiments. "
                           "Only dataset samples (Social) are generated. "
                           "It runs the 4 social cue modalities for Coutrot2. "}


@InferenceConfigRegistrar.register
class InferGeneratorAllModelsDIEM(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DIEMSampleReader"
    sampler = "DIEMSample"

    write_images = True  # if only realtime capture is False

    @classmethod
    def config_info(cls):
        return {"summary": "This generates the datasets images needed for a majority of the experiments. "
                           "Only dataset samples (Social) are generated. "
                           "It runs the 4 social cue modalities for DIEM. "}


@InferenceConfigRegistrar.register
class InferGeneratorFindWho(InferGeneratorAllModelsBase):
    # define the reader
    reader = "FindWhoSampleReader"
    sampler = "FindWhoSample"

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}
    model_groups = [
        [
            ["DAVEInference", 16, [15], {}, {},
             dict(postproc_properties, **{
                 "plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_dave"]]})],
            ["ESR9Inference", 16, [15], {}, {},
             dict(postproc_properties, **{"plot_override": [["det_source_esr9", "det_transformed_esr9"]]})],
            ["Gaze360Inference", 7, [3], {}, {},
             dict(postproc_properties, **{"plot_override": [["det_transformed_gaze360"]]})],
            # ["VideoGazeInference", 7, [3], {}, {},
            #  dict(postproc_properties, **{"plot_override": [["det_transformed_vidgaze"]]})]
        ],
    ]
    width, height = 512, 320  # the frame's width and height
    visualize_images = True  # visualize the plotters
    write_images = True  # if only realtime capture is False
    write_videos = False  # if only realtime capture is False
    write_annotations = True  # always set to False, since annotations not needed for training the models
    process_dataset_videos_only = False  # process videos only if they exist in the train,val,test sets if only realtime capture is False

    @classmethod
    def config_info(cls):
        return {"summary": "This generates the datasets annotation needed for gaze prediction experiments. "
                           "Only dataset samples (Social) are generated. "
                           "It runs the DAVE for FindWhos. "}


@InferenceConfigRegistrar.register
class InferMetricsGASP(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": False, "color_map": "bone",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["GASPInference", 16, [15], {},
          {"inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                  "det_transformed_vidgaze", "det_transformed_gaze360"]},
          dict(postproc_properties, **{
              "plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_gasp"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_gasp",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultgazenet.csv"

    datasplitter_properties = {"train_csv_file": None,
                               "val_csv_file": None,
                               "test_csv_file": "datasets/processed/test_ave.csv"}

    process_dataset_videos_only = True
    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on GASP. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsGASPTrain(InferMetricsGASP):

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}
    model_groups = [
        [["GASPInference", -1, -1, {}, {"inp_img_names_list": None},
          dict(postproc_properties, **{
              "plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_gasp"]]})]]
    ]


@InferenceConfigRegistrar.register
class InferMetricsSTAViS(InferGeneratorAllModelsBase):

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["STAViSInference", 16, [15], {"audiovisual": True}, {},
          dict(postproc_properties, **{"plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_stavis"]]})],
         ["ESR9Inference", 16, [15], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_source_esr9", "det_transformed_esr9"]]})],
         ["Gaze360Inference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_gaze360"]]})],
         ["VideoGazeInference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_vidgaze"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_stavis",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultstavis.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on STAViS. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsSTAViS_VisOnly(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": False, "color_map": "bone",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [("STAViSInference", 16, [15], {"audiovisual": False}, {},
          dict(postproc_properties, **{
              "plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_stavis"]]}))]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_stavis",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultstavis_vis.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on STAViS. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsDAVE(InferGeneratorAllModelsBase):

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["DAVEInference", 16, [15], {}, {},
          dict(postproc_properties, **{
              "plot_override": [["captured", "transformed_salmap", "transformed_fixmap", "det_transformed_dave"]]})],
         ["ESR9Inference", 16, [15], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_source_esr9", "det_transformed_esr9"]]})],
         ["Gaze360Inference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_gaze360"]]})],
         ["VideoGazeInference", 7, [3], {}, {},
          dict(postproc_properties, **{"plot_override": [["det_transformed_vidgaze"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_dave",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultdave.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on DAVE. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsDAVE_VisOnly(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": False, "color_map": "bone",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["DAVEInference", 16, [15], {}, {},
          dict(postproc_properties, **{
              "plot_override": [["captured",
                                 "transformed_salmap",
                                 "transformed_fixmap",
                                 "det_transformed_dave"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_dave",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultdave_vis.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on DAVE. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsTASED_VisOnly(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": False, "color_map": "bone",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["TASEDInference", 32, [31], {}, {},
          dict(postproc_properties, **{"plot_override": [["captured",
                                                          "transformed_salmap",
                                                          "transformed_fixmap",
                                                          "det_transformed_tased"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_tased",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaulttased_vis.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on TASED. "
                           "Only dataset samples (Social) are generated. "}


@InferenceConfigRegistrar.register
class InferMetricsUNISAL_VisOnly(InferGeneratorAllModelsBase):
    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": False, "color_map": "bone",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap"]}

    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    model_groups = [
        [["UNISALInference", 12, [11], {}, {},
             dict(postproc_properties, **{"plot_override": [["captured",
                                                             "transformed_salmap",
                                                             "transformed_fixmap",
                                                             "det_transformed_unisal"]]})]]
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays": False, "color_map": "bone"}

    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[
                                                                               1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    compute_metrics = True
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_unisal",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/defaultunisals_vis.csv"

    @classmethod
    def config_info(cls):
        return {"summary": "This measures the saliency metrics on UNISAL. "
                           "Only dataset samples (Social) are generated. "}



@InferenceConfigRegistrar.register
class InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm(object):

    # define the reader
    reader = "DataSampleReader"
    sampler = "DataSample"

    reader_properties = {}
    sampler_properties = {}
    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                          "enable_transform_overlays": True, "color_map": "jet",
                          "img_names_list": ["transformed_salmap", "transformed_fixmap",
                                             "det_transformed_dave",
                                             "det_transformed_esr9",
                                             "det_transformed_vidgaze",
                                             "det_transformed_gaze360"]}

    # define the face detector
    face_detector = "SFDFaceDetection"  # "MTCNNFaceDetection", "DlibFaceDetection"
    # define audio features needed by the models
    audio_features = {"audio_features": "MFCCAudioFeatures",
                      "hann_audio_frames": "WindowedAudioFeatures"}

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    #  model_name, window_size, source_frames_idxs, enable_audio, preproc_properties, postproc_properties ->
    model_groups = [
        [
            # ["GASPInference", 10, [9], {"weights_file": "seqdamalstmgmu_110nofer", "modalities": 4, "batch_size": 1, "sequence_len": 10, "sequence_norm": True},
            ["GASPInference", 10, [9], {"weights_file": "seqdamalstmgmu", "modalities": 5, "batch_size": 1, "sequence_len": 10, "sequence_norm": True},
            # ["GASPInference", 1, [0], {"modalities": 5, "batch_size": 1, "model_name": "GASPDAMEncGMUConv", "frames_len": 1, "weights_file": "damgmu"},
            {"inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                 "det_transformed_vidgaze", "det_transformed_gaze360"]},
             dict(postproc_properties, **{"plot_override": [["transformed_fixmap",
                                                             "det_transformed_esr9",
                                                             "det_transformed_dave",
                                                             "det_transformed_vidgaze",
                                                             "det_transformed_gaze360",
                                                             "det_transformed_gasp"]]})]
        ],
    ]

    inference_properties = {"show_det_saliency_map": True, "enable_transform_overlays":True, "color_map": "jet"}
    # inference_properties = {"show_saliency_map": True}

    # define the metrics calculator
    metrics = "SaliencyPredictionMetrics"
    metrics_list = ["aucj", "aucs", "cc", "nss", "sim"]
    metrics_mappings = {"gt_salmap": "transformed_salmap",
                        "gt_fixmap": "transformed_fixmap",
                        "pred_salmap": "det_transformed_dave",
                        "gt_baseline": "datasets/processed/center_bias_bw.jpg",  # "gt_baseline": "transformed_fixmap"
                        "scores_info": ["gate_scores"]}
    metrics_save_file = "logs/metrics/default.csv"

    datasplitter_properties = {"train_csv_file": "datasets/processed/test_ave.csv",
                               "val_csv_file": None,
                               "test_csv_file": None}
    # constants
    width, height = 500, 500  # the frame's width and height
    stride = 1  # the number of frames to capture per inference iteration. Should be lte than max_w_size
    max_w_size = 10  # AUTO: the largest window needed by any model
    enable_audio = True  # AUTO: if only one of the models needs audio, then this will automatically be True
    play_audio = False  # if any of the models employing audio has no source_frames_idxs. Check keep_audio in postproc_properties
    realtime_capture = False  # capture audio/video in realtime (cam/mic)
    visualize_images = True  # visualize the plotters
    write_images = False  # if only realtime capture is False
    write_videos = True  # if only realtime capture is False
    write_annotations = False  # always set to False, since annotations not needed for training the models
    process_dataset_videos_only = True  # process videos only if they exist in the train,val,test sets if only realtime capture is False
    compute_metrics = False  # enable the metrics computation
    device = "cpu"  # the pytorch device to use for all models
    n_jobs = [len(model_groups[0]) + 2] + [len(model_group) for model_group in model_groups[1:]]  # AUTO: number of jobs to run in parallel per model group. Extraction in group[0]

    @classmethod
    def config_info(cls):
        return {"summary": "This visualizes the sequential GASP model (DAM + LARGMU; Context Size = 10)"}


@InferenceConfigRegistrar.register
class InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm_110(InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm):
    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": True, "color_map": "jet",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap",
                                              "det_transformed_dave",
                                              "det_transformed_vidgaze",
                                              "det_transformed_gaze360"]}

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    #  model_name, window_size, source_frames_idxs, enable_audio, preproc_properties, postproc_properties ->
    model_groups = [
        [
            ["GASPInference", 10, [9], {"weights_file": "seqdamalstmgmu_110nofer", "modalities": 4, "batch_size": 1, "sequence_len": 10, "sequence_norm": True},
             {"inp_img_names_list": ["captured", "det_transformed_dave",
                                     "det_transformed_vidgaze", "det_transformed_gaze360"]},
             dict(postproc_properties, **{"plot_override": [["transformed_fixmap",
                                                             "det_transformed_dave",
                                                             "det_transformed_vidgaze",
                                                             "det_transformed_gaze360",
                                                             "det_transformed_gasp"]]})]
        ],
    ]

    # constants
    max_w_size = 10  # AUTO: the largest window needed by any model

    @classmethod
    def config_info(cls):
        return {"summary": "This visualizes the sequential GASP model (DAM + LARGMU; Context Size = 10) "
                           "excluding the FER modality"}


@InferenceConfigRegistrar.register
class InferVisualizeGASPDAMGMU1x1Conv(InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm):
    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": True, "color_map": "jet",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap",
                                              "det_transformed_dave",
                                              "det_transformed_esr9",
                                              "det_transformed_vidgaze",
                                              "det_transformed_gaze360"]}

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    #  model_name, window_size, source_frames_idxs, enable_audio, preproc_properties, postproc_properties ->
    model_groups = [
        [
            ["GASPInference", 1, [0], {"weights_file": "damgmu", "modalities": 5, "batch_size": 1, "model_name": "GASPDAMEncGMUConv", "frames_len": 1},
             {"inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                     "det_transformed_vidgaze", "det_transformed_gaze360"]},
             dict(postproc_properties, **{"plot_override": [["transformed_fixmap",
                                                             "det_transformed_dave",
                                                             "det_transformed_esr9",
                                                             "det_transformed_vidgaze",
                                                             "det_transformed_gaze360",
                                                             "det_transformed_gasp"]]})]
        ],
    ]

    # constants
    max_w_size = 1  # AUTO: the largest window needed by any model

    @classmethod
    def config_info(cls):
        return {"summary": "This visualizes the static GASP model (DAM + GMU)"}


@InferenceConfigRegistrar.register
class InferVisualizeGASPSeqDAMGMUALSTM1x1Conv_10Norm(InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm):
    sampling_properties = {"show_fixation_locations": True, "show_saliency_map": True,
                           "enable_transform_overlays": True, "color_map": "jet",
                           "img_names_list": ["transformed_salmap", "transformed_fixmap",
                                              "det_transformed_dave",
                                              "det_transformed_esr9",
                                              "det_transformed_vidgaze",
                                              "det_transformed_gaze360"]}

    # define the models
    postproc_properties = {"keep_properties": False, "keep_audio": False,
                           "keep_plot_frames_only": True, "resize_frames": True}

    #  model_name, window_size, source_frames_idxs, enable_audio, preproc_properties, postproc_properties ->
    model_groups = [
        [
            ["GASPInference", 10, [9], {"weights_file": "seqdamgmualstm", "modalities": 5, "batch_size": 1, "model_name": "SequenceGASPDAMEncGMUALSTMConv", "sequence_len": 10, "sequence_norm": True},
             {"inp_img_names_list": ["captured", "det_transformed_dave", "det_transformed_esr9",
                                     "det_transformed_vidgaze", "det_transformed_gaze360"]},
             dict(postproc_properties, **{"plot_override": [["transformed_fixmap",
                                                             "det_transformed_dave",
                                                             "det_transformed_esr9",
                                                             "det_transformed_vidgaze",
                                                             "det_transformed_gaze360",
                                                             "det_transformed_gasp"]]})]
        ],
    ]

    # constants
    max_w_size = 10  # AUTO: the largest window needed by any model

    @classmethod
    def config_info(cls):
        return {"summary": "This visualizes the static GASP model (DAM + GMU)"}
