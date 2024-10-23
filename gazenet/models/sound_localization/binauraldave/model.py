#
# DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction
# https://arxiv.org/abs/1905.10693
# https://hrtavakoli.github.io/DAVE/
#
# Copyright by Hamed Rezazadegan Tavakoli
# Modified for binaural support by Fares Abawi (fares.abawi@uni-hamburg.de)

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from torch.utils.data import DataLoader
from torchvision import transforms

from gazenet.utils.registrar import *
from gazenet.utils.helpers import UnNormalize
from gazenet.models.shared_components.resnet3d.model import resnet18
from gazenet.models.sound_localization.binauraldave.generator import *
from gazenet.models.sound_localization.binauraldave.losses import kl_div_loss, cc_score, sim_score

EVAL_VISUALIZATION_PROBABILITY = 0.001
TRAIN_LOG_ON_STEP = True
TRAIN_LOG_ON_EPOCH = False

MODEL_PATHS = {
    "monaural_dave": os.path.join("gazenet", "models", "saliency_prediction", "dave",
                                  "checkpoints", "pretrained_dave_orig", "model.pth.tar")}


class ScaleUp(nn.Module):
    def __init__(self, in_size, out_size, momentum=0.1):
        super(ScaleUp, self).__init__()

        self.combine = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size, momentum=momentum)

        self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.combine.weight)
        nn.init.constant_(self.combine.bias, 0.0)

    def forward(self, inputs):
        output = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.combine(output)
        output = F.relu(output, inplace=True)
        return output


@ModelRegistrar.register
class DAVEBase(pl.LightningModule):

    def __init__(self,  learning_rate=0.000014, batch_size=8, num_workers=16, loss_weights=(1, 0.01, 0.005),
                 frames_len=16, num_classes_video=400, num_classes_audio=12,
                 trg_img_width=40, trg_img_height=32,
                 inp_img_width=320, inp_img_height=256,
                 inp_img_mean=(110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0),
                 inp_img_std=(38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0),
                 momentum=0.1,
                 train_dataset_properties=None, val_dataset_properties=None, test_dataset_properties=None,
                 val_store_image_samples=False, eval_extra_metrics=False,
                 freeze_video_branch=True, freeze_audio_branches=False,
                 load_monaural_dave=True, monaural_dave_weights_file=MODEL_PATHS["monaural_dave"],
                 audiovisual=True, binaural=True, **kwargs):
        super(DAVEBase, self).__init__()
        # the default audio branch. Only used for copying the loaded weights to the left/right audio branches
        self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=frames_len,
                                           num_classes=num_classes_audio, last_fc=False, last_pool=True, momentum=momentum)

        self.audio_branch_right = resnet18(shortcut_type='A', sample_size=64, sample_duration=frames_len,
                                           num_classes=num_classes_audio, last_fc=False, last_pool=True, momentum=momentum)
        self.audio_branch_left = resnet18(shortcut_type='A', sample_size=64, sample_duration=frames_len,
                                          num_classes=num_classes_audio, last_fc=False, last_pool=True, momentum=momentum)
        self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=frames_len,
                                     num_classes=num_classes_video, last_fc=False, last_pool=False, momentum=momentum)
        self.upscale1 = ScaleUp(512, 512, momentum=momentum)
        self.upscale2 = ScaleUp(512, 128, momentum=momentum)
        self.combinedAudioEmbedding = nn.Conv2d(1024, 512, kernel_size=1)
        self.combinedEmbedding = nn.Conv2d(1024, 512, kernel_size=1)
        self.saliency = nn.Conv2d(128, 1, kernel_size=1)
        self._weights_init()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.loss_weights = loss_weights
        self.loss_weights_named = {"kld_loss": 1 / loss_weights[0] if loss_weights[0] != 0 else 0,
                                   "cc": -1 / loss_weights[1] if loss_weights[1] != 0 else 0,
                                   "sim": -1 / loss_weights[2] if loss_weights[2] != 0 else 0,
                                   "loss": 1}
        # if eval_extra_metrics:
        #     self.loss_weights_named.update(**{
        #         "sim": 1})

        self.frames_len = frames_len
        self.trg_img_width = trg_img_width
        self.trg_img_height = trg_img_height
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        self.train_dataset_properties = train_dataset_properties
        self.val_dataset_properties = val_dataset_properties
        self.test_dataset_properties = test_dataset_properties

        self.val_store_image_samples = val_store_image_samples
        self.eval_extra_metrics = eval_extra_metrics

        if audiovisual and load_monaural_dave:
            if monaural_dave_weights_file in MODEL_PATHS.keys():
                monaural_dave_weights_file = MODEL_PATHS[monaural_dave_weights_file]
            self.load_monaural_model(weights_file=monaural_dave_weights_file)
            self.audio_branch_left.load_state_dict(self.audio_branch.state_dict())
            self.audio_branch_right.load_state_dict(self.audio_branch.state_dict())

        # delete the default audio branch after loading completed
        del self.audio_branch

        # freezing layers
        if freeze_video_branch:
            for p in self.video_branch.parameters():
                p.requires_grad = False
        if freeze_audio_branches:
            for p in self.audio_branch_left.parameters():
                p.requires_grad = False
            for p in self.audio_branch_right.parameters():
                p.requires_grad = False

        self.audiovisual = audiovisual
        self.binaural = binaural

    def load_model(self, weights_file, device=None):
        if device is None:
            self.load_state_dict(torch.load(weights_file))
        else:
            self.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    def load_monaural_model(self, weights_file, device=None):
        self.load_state_dict(self._load_state_dict_(weights_file, device), strict=False)

    @staticmethod
    def _load_state_dict_(weights_file, device=None):
        if os.path.isfile(weights_file):
            # print("=> loading checkpoint '{}'".format(filepath))
            if device is None:
                checkpoint = torch.load(weights_file)
            else:
                checkpoint = torch.load(weights_file, map_location=torch.device(device))
            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            return state_dict

    def _weights_init(self):

        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

        nn.init.kaiming_normal_(self.combinedEmbedding.weight)
        nn.init.constant_(self.combinedEmbedding.bias, 0.0)

        nn.init.kaiming_normal_(self.combinedAudioEmbedding.weight)
        nn.init.constant_(self.combinedAudioEmbedding.bias, 0.0)

    def loss(self, logits, y, extra_metrics=False):
        kld_loss = kl_div_loss(logits, y["transformed_audiomap"], self.loss_weights[0])
        cc_loss = cc_score(logits, y["transformed_audiomap"], self.loss_weights[1])
        sim_loss = sim_score(logits, y["transformed_audiomap"], self.loss_weights[2])

        loss_returns = {"kld_loss": kld_loss, "cc": cc_loss, "sim": sim_loss, "loss": kld_loss + cc_loss + sim_loss}
        return loss_returns

    def training_step(self, train_batch, batch_idx):
        imgs, aud_left, aud_right, gt = train_batch
        logits = self.forward(imgs,
                              aud_left if self.audiovisual else None,
                              aud_right if (self.audiovisual and self.binaural) else None)
        losses = self.loss(logits, gt)
        logs = {f'train_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=TRAIN_LOG_ON_STEP, on_epoch=TRAIN_LOG_ON_EPOCH)
        return {"loss": losses["loss"]} # "log": logs

    def validation_step(self, val_batch, batch_idx):
        imgs, aud_left, aud_right, gt = val_batch
        logits = self.forward(imgs,
                              aud_left if self.audiovisual else None,
                              aud_right if (self.audiovisual and self.binaural) else None)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'val_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"val_loss": logs["val_loss"]}

    def test_step(self, test_batch, batch_idx):
        imgs, aud_left, aud_right, gt = test_batch
        logits = self.forward(imgs,
                              aud_left if self.audiovisual else None,
                              aud_right if (self.audiovisual and self.binaural) else None)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'test_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"test_loss": logs["test_loss"]}

    def training_epoch_end(self, outputs):
        if "log" in outputs[0].keys():
            for log_key in outputs[0]['log'].keys():
                self.log(f'avg_{log_key}', torch.stack([output['log'][log_key] for output in outputs]).mean())
                self.log(f'std_{log_key}', torch.stack([output['log'][log_key] for output in outputs]).std())
        return None

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        for log_key in outputs[0].keys():
            self.log(f'avg_{log_key}', torch.stack([output[log_key] for output in outputs]).mean())
            self.log(f'std_{log_key}', torch.stack([output[log_key] for output in outputs]).std())

    def test_epoch_end(self, outputs):
        if outputs:
            for log_key in outputs[0].keys():
                self.log(f'avg_{log_key}', torch.stack([output[log_key] for output in outputs]).mean())
                self.log(f'std_{log_key}', torch.stack([output[log_key] for output in outputs]).std())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):

        # transforms for images
        input_img_transform = transforms.Compose([
            transforms.Resize((self.inp_img_height, self.inp_img_width)),
            transforms.ToTensor(),
            transforms.Normalize(self.inp_img_mean, self.inp_img_std),
        ])
        gt_img_transform = transforms.Compose([
            transforms.Resize((self.trg_img_height, self.trg_img_width)),
            transforms.ToTensor()
        ])

        self.train_dataset_properties.update(frames_len=self.frames_len, gt_img_transform=gt_img_transform,
                                             inp_img_transform=input_img_transform)
        self.train_dataset = BinauralDAVEDataset(**self.train_dataset_properties, audiovisual=self.audiovisual)

        self.val_dataset_properties.update(frames_len=self.frames_len, gt_img_transform=gt_img_transform,
                                           inp_img_transform=input_img_transform, random_flips=False)
        self.val_dataset = BinauralDAVEDataset(**self.val_dataset_properties, audiovisual=self.audiovisual)

        self.test_dataset_properties.update(frames_len=self.frames_len, gt_img_transform=gt_img_transform,
                                            inp_img_transform=input_img_transform, random_flips=False)
        self.test_dataset = BinauralDAVEDataset(**self.test_dataset_properties, audiovisual=self.audiovisual)

        # self.train_dataset, self.val_dataset = random_split(self.train_dataset, [55000, 5000])
        # update the batch size based on the train dataset if experimenting on small dummy data
        if len(self.train_dataset) < self.batch_size:
            self.batch_size = len(self.train_dataset)

    def log_val_images(self, x, logits, y, step):
        if isinstance(self.logger, CometLogger) or isinstance(self.logger, WandbLogger):
            # for 1 channel plotting
            from PIL import Image
            import numpy as np
            from matplotlib import cm

            random_batch_item_idx = np.random.randint(0, max(x.shape[0] - 1, 1)) if self.batch_size > 1 else 0

            # the gt
            imgs = []
            gt = np.squeeze(y["transformed_audiomap"].cpu()[random_batch_item_idx, 0, ::].data.numpy())
            imgs.append(Image.fromarray(np.uint8(cm.jet(gt) * 255)).resize((self.inp_img_width, self.inp_img_height)))

            # the input
            for seq_idx in (0, self.frames_len-1):
                x_mod = UnNormalize(self.inp_img_mean, self.inp_img_std)(x[random_batch_item_idx, :3, seq_idx, ::])
                DEBUG_x = np.squeeze(x_mod.cpu().data.numpy())
                DEBUG_x = np.moveaxis(DEBUG_x, 0, -1)  # change 0:2 to the input channels of the modality -> first modality 0:3
                imgs.append(Image.fromarray((DEBUG_x * 255).astype(np.uint8)))

            # the prediction
            pred = logits.cpu()[random_batch_item_idx, ::]
            pred_norm = pred - torch.min(pred)
            pred_norm /= torch.max(pred_norm)
            DEBUG_y = np.squeeze(pred_norm.data.numpy())
            DEBUG_y = np.uint8(cm.jet(DEBUG_y) * 255)
            imgs.append(Image.fromarray(DEBUG_y).resize((self.inp_img_width, self.inp_img_height)))

            # collate images
            widths, heights = zip(*(i.size for i in imgs))

            total_width = sum(widths) // 4
            max_height = max(heights)
            total_height = max_height * 4

            new_im = Image.new('RGB', (total_width, total_height))

            y_offset = 0
            for im_y_idx in range(0, 4):
                new_im.paste(imgs[im_y_idx], (0, y_offset))
                y_offset += imgs[im_y_idx].size[1]

            img_key = str(step) + "_" + str(random_batch_item_idx)
            if isinstance(self.logger, CometLogger):
                # log on comet_ml
                self.logger.experiment.log_image(new_im, 'GT : INPUTS : PRED :: ' + img_key, step=self.current_epoch)
            if isinstance(self.logger, WandbLogger):
                # log on wandb
                self.logger.log_image(images=[new_im], key=img_key, caption=['GT : INPUTS : PRED'], step=self.current_epoch)


    @staticmethod
    def update_infer_config(log_path, checkpoint_file, train_config, infer_config, device):
        # if isinstance(device, list):
        #     infer_config.device = "cuda:" + str(device[0])
        if isinstance(device, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device)
        elif isinstance(device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        # TODO (fabawi): the last checkpoint isn't always the best but this is the safest option for now
        # update the datasplitter to include test file specified in the trainer
        infer_config.datasplitter_properties.update(test_csv_file=train_config.test_dataset_properties["csv_file"])
        # update the metrics file path
        infer_config.metrics_save_file = os.path.join(log_path, "metrics.csv")

        # update the targeted model properties
        for mod_group in infer_config.model_groups:
            for mod in mod_group:
                if not train_config.inferer_name == mod[0]:
                    continue

                # try extracting the window size if it exists, otherwise, assume single frame
                for w_size_name in ["w_size", "frames_len"]:
                    if w_size_name in train_config.model_properties:
                        if mod[1] == -1:
                            mod[1] = train_config.model_properties[w_size_name]
                        if mod[2] == -1:
                            mod[2] = [mod[1] - 1]
                        break
                if mod[1] == -1:
                    mod[1] = 1
                if mod[2] == -1:
                    mod[2] = [0]

                # infer the frames_len from sequence_len
                if "frames_len" not in train_config.model_properties and "sequence_len" in train_config.model_properties:
                    train_config.model_properties["frames_len"] = train_config.model_properties["sequence_len"]

                # update the configuration
                mod[3].update(**train_config.model_properties,
                              weights_file=checkpoint_file,
                              model_name=train_config.model_name)

                break

        return infer_config


@ModelRegistrar.register
class BinauralDAVE(DAVEBase):
    def __init__(self, *args, **kwargs):
        super(BinauralDAVE, self).__init__(*args, audiovisual=True, binaural=True, **kwargs)

    def forward(self, v, a_left=None, a_right=None, return_latent_streams=False):
        # V video video_frames_list of 3x16x256x320
        # A audio video_frames_list of 3x16x64x64
        # return a map of 32x40

        xV1 = self.video_branch(v)

        if a_left is not None:
            xA1 = self.audio_branch_left(a_left)
            xA1 = xA1.expand_as(xV1)
        else:
            # replace left audio branch with zeros
            xA1 = torch.zeros_like(xV1)
        if a_right is not None:
            xA2 = self.audio_branch_right(a_right)
            xA2 = xA2.expand_as(xV1)
        else:
            # replace right audio branch with zeros
            xA2 = torch.zeros_like(xV1)

        xCA = torch.cat((xA1, xA2), dim=1)
        xCA = torch.squeeze(xCA, dim=2)
        xA = self.combinedAudioEmbedding(xCA)
        xA = torch.unsqueeze(xA, dim=2)

        xC = torch.cat((xV1, xA), dim=1)
        xC = torch.squeeze(xC, dim=2)
        x = self.combinedEmbedding(xC)
        x = F.relu(x, inplace=True)

        x = torch.squeeze(x, dim=2)
        x = self.upscale1(x)
        x = self.upscale2(x)
        sal = self.saliency(x)
        sal = F.relu(sal, inplace=True)
        if return_latent_streams:
            return sal, xV1, xA1, xA2
        else:
            return sal


@ModelRegistrar.register
class MonauralDAVE(DAVEBase):
    def __init__(self, *args, **kwargs):
        super(MonauralDAVE, self).__init__(*args, audiovisual=True, binaural=False, **kwargs)

    def training_step(self, train_batch, batch_idx):
        imgs, aud, _, gt = train_batch
        logits = self.forward(imgs, aud)
        losses = self.loss(logits, gt)
        logs = {f'train_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=TRAIN_LOG_ON_STEP, on_epoch=TRAIN_LOG_ON_EPOCH)
        return {"loss": losses["loss"]} # "log": logs

    def validation_step(self, val_batch, batch_idx):
        imgs, aud, _, gt = val_batch
        logits = self.forward(imgs, aud)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'val_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"val_loss": logs["val_loss"]}

    def test_step(self, test_batch, batch_idx):
        imgs, aud, _, gt = test_batch
        logits = self.forward(imgs, aud)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'test_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"test_loss": logs["test_loss"]}

    def forward(self, v, a, return_latent_streams=False):
        # V video video_frames_list of 3x16x256x320
        # A audio video_frames_list of 3x16x64x64
        # return a map of 32x40

        xV1 = self.video_branch(v)

        if a is not None:
            xA1 = self.audio_branch_left(a)
            xA1 = xA1.expand_as(xV1)
        else:
            # replace left audio branch with zeros
            xA1 = torch.zeros_like(xV1)

        xC = torch.cat((xV1, xA1), dim=1)
        xC = torch.squeeze(xC, dim=2)
        x = self.combinedEmbedding(xC)
        x = F.relu(x, inplace=True)

        x = torch.squeeze(x, dim=2)
        x = self.upscale1(x)
        x = self.upscale2(x)
        sal = self.saliency(x)
        sal = F.relu(sal, inplace=True)
        if return_latent_streams:
            return sal, xV1, xA1
        else:
            return sal


@ModelRegistrar.register
class VisDAVE(DAVEBase):
    def __init__(self, *args, **kwargs):
        super(VisDAVE, self).__init__(*args, audiovisual=False, binaural=False, **kwargs)

    def training_step(self, train_batch, batch_idx):
        imgs, gt = train_batch
        logits = self.forward(imgs)
        losses = self.loss(logits, gt)
        logs = {f'train_{k}': (v * self.loss_weights_named[k]) / self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=TRAIN_LOG_ON_STEP, on_epoch=TRAIN_LOG_ON_EPOCH)
        return {"loss": losses["loss"]}  # "log": logs

    def validation_step(self, val_batch, batch_idx):
        imgs, gt = val_batch
        logits = self.forward(imgs)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'val_{k}': (v * self.loss_weights_named[k]) / self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"val_loss": logs["val_loss"]}

    def test_step(self, test_batch, batch_idx):
        imgs, gt = test_batch
        logits = self.forward(imgs)

        if random.random() < EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(imgs, logits, gt, step=batch_idx)
        losses = self.loss(logits, gt, extra_metrics=self.eval_extra_metrics)
        logs = {f'test_{k}': (v * self.loss_weights_named[k]) / self.batch_size for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        return {"test_loss": logs["test_loss"]}

    def forward(self, v, return_latent_streams=False):
        # V video video_frames_list of 3x16x256x320
        # A audio video_frames_list of 3x16x64x64
        # return a map of 32x40

        xV1 = self.video_branch(v)
        xA1 = torch.zeros_like(xV1)
        xC = torch.cat((xV1, xA1), dim=1)
        xC = torch.squeeze(xC, dim=2)
        x = self.combinedEmbedding(xC)
        x = F.relu(x, inplace=True)

        x = torch.squeeze(x, dim=2)
        x = self.upscale1(x)
        x = self.upscale2(x)
        sal = self.saliency(x)
        sal = F.relu(sal, inplace=True)
        if return_latent_streams:
            return sal, xV1
        else:
            return sal
