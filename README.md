# GASP: Gated Attention for Saliency Prediction

[\[Project Page: KT\]](http://software.knowledge-technology.info/#gasp) | [\[Abstract\]](https://www.ijcai.org/proceedings/2021/81) | [\[Paper\]](https://www.ijcai.org/proceedings/2021/0081.pdf) | [\[Poster\]](showcase/IJCAI_21_Poster.pdf) | [\[BibTeX\]](https://www.ijcai.org/proceedings/2021/bibtex/81) | [\[Video\]](https://www.youtube.com/watch?v=e4HFTmEgirk&t=7s)

This is the official [GASP](http://software.knowledge-technology.info/#gasp) code for our paper, presented at 
[IJCAI 2021](https://ijcai-21.org). If you find this work useful, please cite our [paper](https://www2alt.informatik.uni-hamburg.de/wtm/publications/2021/AWW21/index.php):

```
@InProceedings{AWW21,
  author       = "Abawi, Fares and Weber, Tom and Wermter, Stefan",
  title        = "{GASP: Gated Attention for Saliency Prediction}",
  booktitle    = "{Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21)}",
  pages        = "584--591",
  month        = "Aug",
  year         = "2021",
  publisher    = "{International Joint Conferences on Artificial Intelligence}",
  organization = "{IJCAI Organization}",
  key          = "abawi2021gasp",
  doi          = "10.24963/ijcai.2021/81",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2021/AWW21/Abawi_IJCAI21.pdf"
}
```

## Architecture Overview

![GASP model architecture showing the feature extraction pipeline in the first stage followed by the feature integration in the second stage](showcase/multimodalsaliency.png)

## Environment Variables and Preparation

Configuring the training and inference pipelines is done through python scripts to maintain flexibility 
in adding functionality within the configuration itself. Configuration can also be added externally in the
form of `json` files found in [infer_configs](gazenet/configs/infer_configs) and [train_configs](gazenet/configs/train_configs). 
In general, all the configurations for repeating the paper experiments 
could be found in [infer_config.py](gazenet/configs/infer_config.py) and [train_config.py](gazenet/configs/train_config.py).

We recommend using [comet.ml](https://comet.ml) for tracking your experiments during training. You can opt to use `tensorboard` as well,
but the logger needs to be specified as an argument `--logger_name tensorboard` to `gasp_train` or changed in the training configuration itself 

`
logger = 'tensorboard' # logger = '' -> does not log the experiment
`

When choosing to use comet.ml by specifying `--logger_name comet`, set the following environment variables:

```
export COMET_WORKSPACE=<YOUR COMET WORKSPACE>
export COMET_KEY=<YOUR COMET API KEY>
```

replacing `<YOUR COMET WORKSPACE>` with your workspace name and `<YOUR COMET API KEY>` with your comet.ml API key.

It is recommended to create a separate working space outside of this repository. This can be done by setting:

```
export GASP_CWD=<DIRECTORY WHERE MODELS, DATASETS AND RESULTS ARE STORED>
# CAN BE SET TO $(pwd) TO INSTALL IN THE CODE REPO:
# export GASP_CWD=$(pwd)
```

## Setup

The following need to be installed:

```
sudo apt-get install pv jq libportaudio2 ffmpeg
```

GASP is implemented in Pytorch and trained using the PytorchLightning library. To install GASP requirements, create a virtual environment and run:

`
python3 setup.py install
` 

### Preprocessing and Generating Social Cues + Saliency Prediction Representations (SCD) 

| <span style="display: inline-block; width:500px"> Saliency Prediction (DAVE) + Video </span> | Gaze Direction Estimation (Gaze360) | Gaze Following (VideoGaze) | Facial Expression Recognition (ESR9) |
| ------------------------- | ------------------------- | ------------------------- |:------------------------- |
| <img src="showcase/det_transformed_dave_coutrot1_clip48_compressed.gif" width="200" height="200"/> | <img src="showcase/det_transformed_gaze360_coutrot1_clip48_compressed.gif" width="200" height="200"/> | <img src="showcase/det_transformed_videogaze_coutrot1_clip48_compressed.gif" width="200" height="200"/> | <img src="showcase/det_transformed_esr9_coutrot1_clip48_compressed.gif" width="200" height="200"/> |   

**Download**
 
You can download the preprocessed spatiotemporal maps directly without the need to process the training data locally:
    
```
gasp_download_manager --working_dir $GASP_CWD \
                      --datasets processed/Grouped_frames/coutrot1 \
                                 processed/Grouped_frames/coutrot2 \
                                 processed/Grouped_frames/diem
```
    

**Preprocess Locally**

Alternatively, generate the modality representations using the provided scripts. 
Note that this might take upwards of a day depending on your CPU and/or GPU. 

1. Download the datasets and pretrained social cue parameters directly by running the following script
 (shells bash scripts and has been tested on Ubuntu 20.4):

    ```
    gasp_download_manager --working_dir $GASP_CWD \
                          --datasets ave/database1 ave/database2 ave/diem stavis_preprocessed \
                          --models emotion_recognition/esr9/checkpoints/pretrained_esr9_orig \
                                   gaze_estimation/gaze360/checkpoints/pretrained_gaze360_orig \
                                   gaze_following/videogaze/checkpoints/pretrained_videogaze_orig \
                                   saliency_prediction/dave/checkpoints/pretrained_dave_orig
    ```
   
    *Note*: You can instead navigate to `datasets/<CHOSEN DATASET>` to download individual datasets and run the corresponding `download_dataset.sh` bash file directly.
    
2. On download completion, the dataset can be generated (run from within the `--working_dir $GASP_CWD` specified in the previous step):

    ```
    gasp_infer --infer_config InferGeneratorAllModelsCoutrot1
    gasp_infer --infer_config InferGeneratorAllModelsCoutrot2
    gasp_infer --infer_config InferGeneratorAllModelsDIEM
    ```
   
3. Finally, you could choose to replace the ground-truth fixation density maps and fixation points by the preprocessed 
   maps generated by [STAViS: Tsiami et al.](https://github.com/atsiami/STAViS) Note that we use these ground-truth maps in all our experiments:

    ```
    gasp_scripts --working_dir $GASP_CWD --scripts postprocess_get_from_stavis
    ```
   
## Training

To train the best achieving sequential model **(DAM + LARGMU; Context Size = 10)**, invoke the script configuration on the social event subset of the [AVE dataset \[Tavakoli et al.\]](https://hrtavakoli.github.io/AVE/):

```
gasp_train --train_config GASPExp002_SeqDAMALSTMGMU1x1Conv_10Norm \
           --infer_configs InferMetricsGASPTrain \
           --checkpoint_save_every_n_epoch 50 --checkpoint_save_n_top 5 --check_val_every_n_epoch 49 --max_epochs 2000 \
           --gpus "0," --logger_name "comet" --val_store_image_samples --compute_metrics

```

or specify a json configuration file:

```
gasp_train --train_config_file $GASP_CWD/gazenet/configs/train_configs/GASPExp002_SeqDAMALSTMGMU1x1Conv_10Norm.json \
           --infer_config_files $GASP_CWD/gazenet/configs/infer_configs/InferMetricsGASPTrain.json \
           --checkpoint_save_every_n_epoch 50 --checkpoint_save_n_top 5 --check_val_every_n_epoch 49 --max_epochs 2000 \
           --gpus "0," --logger_name "comet" --val_store_image_samples --compute_metrics

```

The `--compute_metrics` argument will run the inference script on completion and store the metrics results in [logs\metrics](logs\metrics) in the working directory.

*Note*: We treat a single peek (covering the context size of GASP) into each of the videos in the dataset as an epoch since we visualize validation samples at short intervals rather than entire epochs. This should not be misconstrued as 2000 epochs over the dataset.

## Inference

![coutrot2 clip13 with sequential GASP DAM + LARGMU (Context Size: 10) overlaid on a video of 4 men in a meeting scenario](showcase/coutrot2_clip13_compressed.gif)

**WARNING: Always run** `gasp_scripts --working_dir $GASP_CWD --scripts clean_temp` **before executing any inference script when changing dataset splits. 
As a precaution, always delete temporary files before executing inference scripts if it doesn't take too long to process.**

The inferer can run and visualize all integrated models as well as the GASP variants. To download all GASP variants:

```
gasp_download_manager --working_dir $GASP_CWD \
                      --models "saliency_prediction/gasp/<...>"
```

To run the GASP inference, select a configuration class or json file and execute the inference script:

```
gasp_infer --infer_config InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm --gpu 0
``` 

*Note*: Remove `--gpu 0` argument to run on CPU.

*Tip*: To try out specific videos, create a new split (.csv files found in [datasets/processed](datasets/processed)) e.g.,:

```
video_id,fps,scene_type,dataset
clip_11,25,Other,coutrot1
```

and in the configuration file e.g., [InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm](gazenet/configs/infer_configs/InferVisualizeGASPSeqDAMALSTMGMU1x1Conv_10Norm.json) replace:

```
"datasplitter_properties": {
        "train_csv_file": "datasets/processed/test_ave.csv",
        "val_csv_file": null,
        "test_csv_file": null
    },
```

by the new split's name:

```
        "train_csv_file": "datasets/processed/<NEW_SPLIT_NAME>.csv"
```

## TODOs

- [ ] Support parallelizing inference models on multiple GPUs
- [ ] Support parallelizing inference models on multiple machines using middleware
- [ ] Support realtime inference for GASP (currently works for selected models)
- [ ] Restructure configuration files for more consistency
- [ ] Support intermediate invocation of external applications within the inference model pipeline

## Attribution

This work relies on several packages and code-bases which we have modified to fit our framework. 
If any attributions are missing, please notify us by [Email](mailto:fares.abawi@uni-hamburg.de?subject=[GitHub]%20Missing%20GASP%20Attribution). 
The following is a list of repositories which have a substantial portion of their content included in this work:

* [STAViS: Spatio-Temporal AudioVisual Saliency Network](https://github.com/atsiami/STAViS)
* [Unified Image and Video Saliency Modeling](https://github.com/rdroste/unisal)
* [TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection](https://github.com/MichiganCOG/TASED-Net)
* [ViNet: Pushing the limits of Visual Modality for Audio-Visual Saliency Prediction](https://github.com/samyak0210/ViNet)
* [DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction](https://github.com/hrtavakoli/DAVE)
* [Gaze360: Physically Unconstrained Gaze Estimation in the Wild Dataset](https://github.com/erkil1452/gaze360)
* [Following Gaze in Video](https://github.com/recasens/Gaze-Following)
* [Efficient Facial Feature Learning with Wide Ensemble-based Convolutional Neural Networks](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks)
* [Saliency Metrics](https://github.com/tarunsharma1/saliency_metrics)

## Acknowledgement

This work was supported by the German
Research Foundation DFG under project [CML (TRR 169)](https://www.crossmodal-learning.org/).
