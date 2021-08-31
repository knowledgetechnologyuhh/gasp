import os

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import *
from torch.utils.data import DataLoader
from torchvision import transforms

from gazenet.utils.registrar import *
from gazenet.models.shared_components.attentive_convlstm.model import AttentiveLSTM, SequenceAttentiveLSTM
from gazenet.models.shared_components.squeezeexcitation.model import SELayer
from gazenet.models.shared_components.gmu.model import GMUConv2d, RGMUConv2d
from gazenet.models.saliency_prediction.gasp.generator import *
from gazenet.models.saliency_prediction.losses import cross_entropy_loss, nss_score, cc_score

LATENT_CONV_C = 32


class DAMLayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(DAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.fc2_sig = nn.Sigmoid()

        self.conv = nn.Sequential(nn.Conv2d(channel, 256, kernel_size=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 1, kernel_size=1))

    def forward(self, x, detached=False):
        if detached:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = torch.nn.functional.linear(y, self.fc1.weight)
            y = self.fc1_relu(y)
            y = torch.nn.functional.linear(y, self.fc2.weight)
            y = self.fc2_sig(y).view(b, c, 1, 1)
            sal = x * y.expand_as(x)
            sal = sal
        else:
            x = torch.log(1 / torch.nn.functional.softmax(x, dim=1))
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc1(y)
            y = self.fc1_relu(y)
            y = self.fc2(y)
            y = self.fc2_sig(y).view(b, c, 1, 1)
            sal = x * y.expand_as(x)
            sal = self.conv(sal)
        return sal


# NOTE (fabawi): for all non-sequential models, the modality encoder is shared amongst modules; for sequential models,
# each modality has its own encoder, but shared across timesteps
class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=LATENT_CONV_C, encoder="Conv"):
        super().__init__()
        self.encoder = encoder

        if encoder == "Deep":
            self.enc = nn.Sequential(

                nn.Conv2d(in_channels, LATENT_CONV_C, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(LATENT_CONV_C, LATENT_CONV_C, kernel_size=3, stride=1, padding=1),

                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(LATENT_CONV_C, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

                # the decoder
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(64, LATENT_CONV_C, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(LATENT_CONV_C, out_channels, kernel_size=3, stride=1, padding=1),
            )
        elif encoder == "Conv":
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, LATENT_CONV_C, kernel_size=3, padding=1),
                nn.Conv2d(LATENT_CONV_C, 64, kernel_size=3, padding=1),

                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),

                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(64, LATENT_CONV_C, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(LATENT_CONV_C, out_channels, kernel_size=3, stride=1, padding=1),
            )

        elif encoder == "MobileNet":
            raise NotImplementedError("MobileNet not yet integrated")

    def forward(self, x):
        x = self.enc(x)
        return x


class GASPBase(pl.LightningModule):

    def __init__(self, learning_rate=0.00014, batch_size=8, num_workers=16, loss_weights=(0.5, 2, 1), dam_loss_weight=0.5,
                 in_channels=3, modalities=4, out_channels=1, sequence_len=1,
                 trg_img_width=60, trg_img_height=60,
                 inp_img_width=120, inp_img_height=120,
                 inp_img_mean=(110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0),
                 inp_img_std=(38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0),
                 train_dataset_properties=None, val_dataset_properties=None, test_dataset_properties=None,
                 val_store_image_samples=False):
        super(GASPBase, self).__init__()

        self.trg_img_width = trg_img_width
        self.trg_img_height = trg_img_height
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        self.dam_loss_weight = dam_loss_weight
        self.loss_weights = loss_weights
        # CAREFUL (fabawi): the following loss_weights_named are signed and inverted
        self.loss_weights_named = {"bce_loss": 1/loss_weights[0],
                                   "cc": -1/loss_weights[1],
                                   "nss": -1/loss_weights[2],
                                   "dam_bce_loss": 1/self.dam_loss_weight,
                                   "loss": 1}
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        # model and dataset mode dependent
        self.in_channels = in_channels
        self.modalities = modalities
        self.out_channels = out_channels
        self.sequence_len = sequence_len
        self.exhaustive = False

        self.train_dataset_properties = train_dataset_properties
        self.val_dataset_properties = val_dataset_properties
        self.test_dataset_properties = test_dataset_properties

        self.val_store_image_samples = val_store_image_samples

    def forward(self, modules):
        raise NotImplementedError("This is the base GASP class and cannot be inherited directly")

    def loss(self, logits, y):
        bce_loss = cross_entropy_loss(logits, y["transformed_salmap"], self.loss_weights[0])
        cc_loss = cc_score(logits, y["transformed_salmap"], self.loss_weights[1])
        nss_loss = nss_score(logits, y["transformed_fixmap"], self.loss_weights[2])
        # the nss and cc are losses here still, but renamed for name matching with the fixed logging values
        return {"bce_loss": bce_loss, "cc": cc_loss, "nss": nss_loss,
                "loss": sum((bce_loss, cc_loss, nss_loss))}
                # "loss": bce_loss}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)

        if len(logits) > 1 and logits[1] is not None:
            # calculate dam loss if model supports it
            if self.exhaustive:
                dam_logit_tgt = "seq_transformed_salmap"
            else:
                dam_logit_tgt = "transformed_salmap"
            dam_logits = torch.nn.functional.binary_cross_entropy_with_logits(logits[1], y[dam_logit_tgt]) * self.dam_loss_weight
        else:
            dam_logits = torch.tensor(0)
        sal_logits = logits[0]
        losses = self.loss(sal_logits, y)
        losses.update(dam_bce_loss=dam_logits)
        logs = {f'train_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        return {'loss': losses['loss'] + dam_logits, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        sal_logits = logits[0]
        if self.val_store_image_samples:
            self.log_val_images(x, logits, y)
        losses = self.loss(sal_logits, y)
        logs = {f'val_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
        return logs

    # TODO (fabawi): if the training is too slow and you don't care about logs, remove this
    def training_epoch_end(self, outputs):
        avg_logs = {}
        for log_key in outputs[0]['log'].keys():
            self.log(f'avg_{log_key}', torch.stack([output['log'][log_key] for output in outputs]).mean())
            # Uncomment the lines below for older lightning versions
            #avg_logs[f'avg_{log_key}'] = torch.stack([output['log'][log_key] for output in outputs]).mean()
        #return{'avg_train_loss': avg_logs['avg_train_loss'], 'log': avg_logs}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_logs = {}
        for log_key in outputs[0].keys():
            self.log(f'avg_{log_key}', torch.stack([output[log_key] for output in outputs]).mean())
            # Uncomment the lines below for older lightning versions
            #avg_logs[f'avg_{log_key}'] = torch.stack([output[log_key] for output in outputs]).mean()
        #return {'avg_val_loss': avg_logs['avg_val_loss'], 'log': avg_logs}

    def prepare_data(self):

        # dataset properties
        if self.train_dataset_properties is None:
            self.train_dataset_properties = {"csv_file": "datasets/processed/train.csv",
                                             "video_dir": "datasets/processed/Grouped_frames",
                                             "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                      "det_transformed_vidgaze",
                                                                      "det_transformed_gaze360"],
                                             "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

        if self.val_dataset_properties is None:
            self.val_dataset_properties = {"csv_file": "datasets/processed/validation.csv",
                                           "video_dir": "datasets/processed/Grouped_frames",
                                           "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                    "det_transformed_vidgaze",
                                                                    "det_transformed_gaze360"],
                                           "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

        if self.test_dataset_properties is None:
            self.test_dataset_properties = {"csv_file": "datasets/processed/test.csv",
                                            "video_dir": "datasets/processed/Grouped_frames",
                                            "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                     "det_transformed_vidgaze",
                                                                     "det_transformed_gaze360"],
                                            "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"]}

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

        self.train_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                            inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.train_dataset = GASPDataset(**self.train_dataset_properties)

        self.val_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                          inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.val_dataset = GASPDataset(**self.val_dataset_properties)

        self.test_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                           inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.test_dataset = GASPDataset(**self.test_dataset_properties)

        # self.train_dataset, self.val_dataset = random_split(self.train_dataset, [55000, 5000])
        # update the batch size based on the train dataset if experimenting on small dummy data
        if len(self.train_dataset) < self.batch_size:
            self.batch_size = len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        return optimizer

    # TODO (fabawi): this should be a callback instead and not part of the model, but will have to do for now.
    def log_val_images(self, x, logits, y):
        if isinstance(self.logger, CometLogger):
            # for 1 channel plotting
            from PIL import Image
            import numpy as np
            from matplotlib import cm

            random_batch_item_idx = np.random.randint(0, max(x.shape[0]-1, 1)) if self.batch_size > 1 else 0
            all_imgs = []


            all_imgs.append([])
            # im.show()

            # the input
            for seq_idx in range(0, max(self.sequence_len, 1)):
                input_imgs = []
                if self.exhaustive:
                    DEBUG_y = np.squeeze(y["seq_transformed_salmap"].cpu()[random_batch_item_idx, seq_idx, ::].data.numpy())
                    input_imgs.append(Image.fromarray(np.uint8(cm.jet(DEBUG_y) * 255)).resize((self.inp_img_width, self.inp_img_height)))
                else:
                    input_imgs.append(Image.new(INP_IMG_MODE, (self.inp_img_width, self.inp_img_height)))
                for mod_idx in range(0, self.modalities*self.in_channels, self.in_channels):
                    if self.sequence_len > 1:
                        x_mod = x[:,seq_idx, ::]
                    else:
                        x_mod = x
                    x_mod = UnNormalize(self.inp_img_mean, self.inp_img_std)(x_mod[random_batch_item_idx, mod_idx:mod_idx+self.in_channels,::])
                    DEBUG_x = np.squeeze(x_mod.cpu().data.numpy())
                    DEBUG_x = np.moveaxis(DEBUG_x, 0, -1) # change 0:2 to the input channels of the modality -> first modality 0:3
                    print(DEBUG_x.shape)
                    im = Image.fromarray((DEBUG_x*255).astype(np.uint8))
                    input_imgs.append(im)

                if seq_idx != max(self.sequence_len, 1) - 1:
                    input_imgs.append(Image.new(INP_IMG_MODE, (self.inp_img_width, self.inp_img_height)))
                all_imgs[-1].extend(input_imgs)
                all_imgs.append([])
                # im.show()
            del all_imgs[-1]

            # the gt
            DEBUG_y = np.squeeze(y["transformed_salmap"].cpu()[random_batch_item_idx, ::].data.numpy())
            im_gt = Image.fromarray(np.uint8(cm.jet(DEBUG_y) * 255)).resize((self.inp_img_width, self.inp_img_height))
            all_imgs[-1][0] = im_gt

            # the model output
            logits_s = logits[0].cpu()
            logits_s = logits_s[random_batch_item_idx, ::]
            # logits_s_norm = torch.sigmoid(logits_s)
            logits_s_norm = logits_s - torch.min(logits_s)
            logits_s_norm /= torch.max(logits_s_norm)
            DEBUG_y = np.squeeze(logits_s_norm.data.numpy())
            DEBUG_y = np.uint8(cm.jet(DEBUG_y) * 255)
            im_pred = Image.fromarray(DEBUG_y).resize((self.inp_img_width, self.inp_img_height))
            all_imgs[-1].append(im_pred)
            # im.show()

            # collate images
            widths, heights = zip(*(z.size for i in all_imgs for z in i))

            total_width = sum(widths) // max(self.sequence_len, 1)
            max_height = max(heights)
            total_height = max_height * max(self.sequence_len, 1)

            new_im = Image.new('RGB', (total_width, total_height))

            y_offset = 0
            for im_y_idx in range(0, max(self.sequence_len, 1)):
                x_offset = 0
                for im_x_idx in range(0, len(all_imgs[0])):
                    new_im.paste(all_imgs[im_y_idx][im_x_idx], (x_offset, y_offset))
                    x_offset += all_imgs[im_y_idx][im_x_idx].size[0]
                y_offset += all_imgs[im_y_idx][0].size[1]

            # log on comet_ml
            self.logger.experiment.log_image(new_im, 'GT : INPUTS : LOGITS', step=self.current_epoch)

            # this section is for debugging only
            logits_s_norm = torch.sigmoid(logits_s)  # 1
            logits_s_norm = logits_s - torch.min(logits_s_norm)
            logits_s_norm /= torch.max(logits_s_norm)
            DEBUG_y = np.squeeze(logits_s_norm.data.numpy())
            DEBUG_y = np.uint8(cm.jet(DEBUG_y) * 255)
            im_pred_sig_norm = Image.fromarray(DEBUG_y).resize((self.inp_img_width, self.inp_img_height))
            logits_s_norm = torch.sigmoid(logits_s)  # 2
            DEBUG_y = np.squeeze(logits_s_norm.data.numpy())
            DEBUG_y = np.uint8(cm.jet(DEBUG_y) * 255)
            im_pred_sig = Image.fromarray(DEBUG_y).resize((self.inp_img_width, self.inp_img_height))
            debug_new_im = Image.new('RGB', (im_pred.size[0] * 3, im_pred.size[1]))
            debug_new_im.paste(im_pred, (0, 0))
            debug_new_im.paste(im_pred_sig, (im_pred.size[0], 0))
            debug_new_im.paste(im_pred_sig_norm, (im_pred.size[0] * 2, 0))
            self.logger.experiment.log_image(debug_new_im, 'PRED : PRED_SIGMOID: PRED_SIGMOID_NORM',
                                             step=self.current_epoch)

    @staticmethod
    def update_infer_config(log_path, checkpoint_file, train_config, infer_config, device):
        # if isinstance(device, list):
        #     infer_config.device = "cuda:" + str(device[0])
        if isinstance(device, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device)
        elif isinstance(device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        # update the datasplitter to include test file specified in the trainer
        infer_config.datasplitter_properties.update(test_csv_file=train_config.test_dataset_properties["csv_file"])
        # update the metrics file path
        infer_config.metrics_save_file = os.path.join(log_path, "metrics.csv")
        # update the sampling names list. This is specific to DataSample
        infer_config.sampling_properties.update(
            img_names_list=train_config.test_dataset_properties["inp_img_names_list"] +
                           train_config.test_dataset_properties["gt_img_names_list"])

        # update the targeted model properties
        for mod_group in infer_config.model_groups:
            for mod in mod_group:
                if not train_config.inferer_name == mod[0]:
                    continue

                # try extracting the window size if it exists, otherwise, assume single frame
                for w_size_name in ["w_size", "frames_len", "sequence_len"]:
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

                # update the input image names list. This is specific to gasp
                if mod[4]["inp_img_names_list"] is None:
                    mod[4].update(inp_img_names_list=train_config.test_dataset_properties["inp_img_names_list"])

                break

        return infer_config


@ModelRegistrar.register
class GASPEncGMUALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncGMUALSTMConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)

        self.att_lstm = AttentiveLSTM(LATENT_CONV_C, LATENT_CONV_C, LATENT_CONV_C, 3, 3)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules

        sal, lateral = self.gmu(fusion)
        sal = self.att_lstm(sal)
        sal = self.saliency_out(sal)
        return sal, None, lateral


@ModelRegistrar.register
class GASPEncALSTMGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncALSTMGMUConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.att_lstm = AttentiveLSTM(LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, 3, 3)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.att_lstm = AttentiveLSTM(in_channels * modalities, in_channels * modalities, in_channels * modalities, 3, 3)
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules
        sal = self.att_lstm(fusion)
        sal, lateral = self.gmu(sal)
        sal = self.saliency_out(sal)
        return sal, None, lateral  # lateral is non-fusion


@ModelRegistrar.register
class GASPEncALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncALSTMConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.att_lstm = AttentiveLSTM(LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, 3, 3)
            self.saliency_out = nn.Conv2d(LATENT_CONV_C * modalities, out_channels, kernel_size=1)
        else:
            self.att_lstm = AttentiveLSTM(in_channels * modalities, in_channels * modalities, in_channels * modalities, 3, 3)
            self.saliency_out = nn.Conv2d(in_channels * modalities, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules
        sal = self.att_lstm(fusion)
        sal = self.saliency_out(sal)
        return sal, None, None


@ModelRegistrar.register
class GASPDAMEncGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPDAMEncGMUConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)
        self.dam = DAMLayer(in_channels*modalities, reduction=2)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        dam_sal = self.dam(modules)
        modules = self.dam(modules, detached=True)

        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels*self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx+self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder

        sal, lateral = self.gmu(fusion)
        sal = self.saliency_out(sal)
        return sal, dam_sal, lateral  # lateral is non-fusion

@ModelRegistrar.register
class GASPDAMEncALSTMGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPDAMEncALSTMGMUConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)
        self.dam = DAMLayer(in_channels*modalities, reduction=2)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.att_lstm = AttentiveLSTM(LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, 3, 3)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.att_lstm = AttentiveLSTM(in_channels * modalities, in_channels * modalities, in_channels * modalities, 3, 3)
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        dam_sal = self.dam(modules)
        modules = self.dam(modules, detached=True)

        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels*self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx+self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder

        sal = self.att_lstm(fusion)
        sal, lateral = self.gmu(sal)
        sal = self.saliency_out(sal)
        return sal, dam_sal, lateral  # lateral is non-fusion


@ModelRegistrar.register
class GASPDAMEncGMUALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPDAMEncGMUALSTMConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=1,
                                                      *args, **kwargs)
        self.dam = DAMLayer(in_channels*modalities, reduction=2)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)

        else:
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.att_lstm = AttentiveLSTM(LATENT_CONV_C, LATENT_CONV_C, LATENT_CONV_C, 3, 3)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        dam_sal = self.dam(modules)
        modules = self.dam(modules, detached=True)

        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels*self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx+self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder

        sal, lateral = self.gmu(fusion)
        sal = self.att_lstm(sal)
        sal = self.saliency_out(sal)
        return sal, dam_sal, lateral


@ModelRegistrar.register
class GASPEncConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncConv, self).__init__(in_channels=in_channels,
                                             modalities=modalities,
                                             out_channels=out_channels,
                                             sequence_len=1,
                                             *args, **kwargs)
        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder)
            self.saliency_in = nn.Conv2d(LATENT_CONV_C * modalities, LATENT_CONV_C, kernel_size=3, padding=1)
        else:
            self.saliency_in = nn.Conv2d(in_channels*modalities, LATENT_CONV_C, kernel_size=3, padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules
        sal = self.saliency_in(fusion)
        sal = self.saliency_out(sal)
        return sal, None, None


@ModelRegistrar.register
class GASPEncAddConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncAddConv, self).__init__(in_channels=in_channels,
                                             modalities=modalities,
                                             out_channels=out_channels,
                                             sequence_len=1,
                                             *args, **kwargs)
        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder)
            self.saliency_in = nn.Conv2d(1, LATENT_CONV_C, kernel_size=3, padding=1)
        else:
            self.saliency_in = nn.Conv2d(1, LATENT_CONV_C, kernel_size=3, padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules

        sal = torch.sum(fusion, dim=1)
        sal = torch.unsqueeze(sal, dim=1)
        sal = self.saliency_in(sal)
        sal = self.saliency_out(sal)
        return sal, None, None


@ModelRegistrar.register
class GASPSEEncConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPSEEncConv, self).__init__(in_channels=in_channels,
                                                   modalities=modalities,
                                                   out_channels=out_channels,
                                                   sequence_len=1,
                                                   *args, **kwargs)

        self.se = SELayer(in_channels*modalities, reduction=2)

        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder)
            self.saliency_in = nn.Conv2d(LATENT_CONV_C * modalities, LATENT_CONV_C, kernel_size=3, padding=1)
        else:
            self.saliency_in = nn.Conv2d(in_channels * modalities, LATENT_CONV_C, kernel_size=3, padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        modules = self.se(modules)
        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels * self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx + self.in_channels, ::]))

            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules
        sal = fusion
        sal = self.saliency_in(sal)
        sal = self.saliency_out(sal)
        return sal, None, None


@ModelRegistrar.register
class SequenceGASPEncALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceGASPEncALSTMConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=sequence_len,
                                                       *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        self.sequence_len = sequence_len
        self.exhaustive = False

        self.encoder = encoder
        if encoder is not None:
            self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
            self.att_lstm = SequenceAttentiveLSTM(LATENT_CONV_C * modalities,
                                                  LATENT_CONV_C * modalities,
                                                  LATENT_CONV_C * modalities,
                                                  3, 3,
                                                  sequence_len=sequence_len,
                                                  sequence_norm=sequence_norm)
            self.saliency_out = nn.Conv2d(LATENT_CONV_C * modalities, out_channels, kernel_size=1)

        else:
            self.att_lstm = SequenceAttentiveLSTM(in_channels * modalities,
                                                  in_channels * modalities,
                                                  in_channels * modalities,
                                                  3, 3,
                                                  sequence_len=sequence_len,
                                                  sequence_norm=sequence_norm)
            self.saliency_out = nn.Conv2d(in_channels*modalities, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for seq_idx in range(0, self.sequence_len):
                mod_fusion=[]
                for mod_idx, mod_step in enumerate(range(0, self.in_channels*self.modalities, self.in_channels)):
                    mod_fusion.append(self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step+self.in_channels, ::]))
                mod_fusion = torch.cat(mod_fusion, 1)
                fusion.append(mod_fusion)
            fusion = torch.stack(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder


        sal = self.att_lstm(fusion)
        sal = self.saliency_out(sal)
        return sal, None, None


@ModelRegistrar.register
class SequenceGASPEncALSTMGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceGASPEncALSTMGMUConv, self).__init__(in_channels=in_channels,
                                                       modalities=modalities,
                                                       out_channels=out_channels,
                                                       sequence_len=sequence_len,
                                                       *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        self.sequence_len = sequence_len
        self.exhaustive = False
        self.encoder = encoder
        if encoder is not None:
            self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
            self.att_lstm = SequenceAttentiveLSTM(LATENT_CONV_C * modalities,
                                                  LATENT_CONV_C * modalities,
                                                  LATENT_CONV_C * modalities,
                                                  3, 3,
                                                  sequence_len=sequence_len,
                                                  sequence_norm=sequence_norm)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.att_lstm = SequenceAttentiveLSTM(in_channels * modalities,
                                                  in_channels * modalities,
                                                  in_channels * modalities,
                                                  3, 3,
                                                  sequence_len=sequence_len,
                                                  sequence_norm=sequence_norm)
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for seq_idx in range(0, self.sequence_len):
                mod_fusion=[]
                for mod_idx, mod_step in enumerate(range(0, self.in_channels*self.modalities, self.in_channels)):
                    mod_fusion.append(self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step+self.in_channels, ::]))
                mod_fusion = torch.cat(mod_fusion, 1)
                fusion.append(mod_fusion)
            fusion = torch.stack(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder

        sal = self.att_lstm(fusion)
        sal, lateral = self.gmu(sal)
        sal = self.saliency_out(sal)
        return sal, None, lateral  # lateral is non-fusion


@ModelRegistrar.register
class SequenceGASPEncGMUALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceGASPEncGMUALSTMConv, self).__init__(in_channels=in_channels,
                                                             modalities=modalities,
                                                             out_channels=out_channels,
                                                             sequence_len=sequence_len,
                                                             *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        self.sequence_len = sequence_len
        self.exhaustive = False
        self.encoder = encoder
        if encoder is not None:
            self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)

        self.att_lstm = SequenceAttentiveLSTM(LATENT_CONV_C, LATENT_CONV_C, LATENT_CONV_C, 3, 3,
                                              sequence_len=sequence_len,
                                              sequence_norm=sequence_norm)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for seq_idx in range(0, self.sequence_len):
                mod_fusion=[]
                for mod_idx, mod_step in enumerate(range(0, self.in_channels*self.modalities, self.in_channels)):
                    mod_fusion.append(self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step+self.in_channels, ::]))
                mod_fusion = torch.cat(mod_fusion, 1)
                mod_fusion, lateral = self.gmu(mod_fusion)
                fusion.append(mod_fusion)
            fusion = torch.stack(fusion, 1)
        else:
            fusion = []  # to operate without an encoder
            for seq_idx in range(0, self.sequence_len):
                mod_fusion, lateral = self.gmu(fusion[:, seq_idx, ::])
                fusion.append(mod_fusion)
            fusion = torch.stack(fusion, 1)
        sal = self.att_lstm(fusion)
        sal = self.saliency_out(sal)
        return sal, None, lateral  # lateral is only the last

    
@ModelRegistrar.register
class SequenceGASPDAMEncGMUALSTMConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceGASPDAMEncGMUALSTMConv, self).__init__(in_channels=in_channels,
                                                             modalities=modalities,
                                                             out_channels=out_channels,
                                                             sequence_len=sequence_len,
                                                             *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        assert encoder is not None, "Encoder must not be None when running sequential DAM"
        self.sequence_len = sequence_len
        self.exhaustive = True
        self.encoder = encoder
        self.dam = DAMLayer(in_channels * modalities, reduction=2)
        self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
        self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.att_lstm = SequenceAttentiveLSTM(LATENT_CONV_C, LATENT_CONV_C, LATENT_CONV_C, 3, 3,
                                              sequence_len=sequence_len,
                                              sequence_norm=sequence_norm)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        fusion = []
        dam_sals = []
        for seq_idx in range(0, self.sequence_len):
            dam_sal = self.dam(modules[:, seq_idx, ::].clone())
            dam_sals.append(dam_sal)
            modules[:, seq_idx, ::] = self.dam(modules[:, seq_idx, ::].clone(), detached=True)
            mod_fusion=[]
            for mod_idx, mod_step in enumerate(range(0, self.in_channels*self.modalities, self.in_channels)):
                mod_fusion.append(self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step+self.in_channels, ::].clone()))
            mod_fusion = torch.cat(mod_fusion, 1)
            mod_fusion, lateral = self.gmu(mod_fusion)
            fusion.append(mod_fusion)
        fusion = torch.stack(fusion, 1)
        dam_sals = torch.stack(dam_sals, 1)

        sal = self.att_lstm(fusion)
        sal = self.saliency_out(sal)
        return sal, dam_sals, lateral  # lateral is only the last


@ModelRegistrar.register
class SequenceGASPDAMEncALSTMGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceGASPDAMEncALSTMGMUConv, self).__init__(in_channels=in_channels,
                                                             modalities=modalities,
                                                             out_channels=out_channels,
                                                             sequence_len=sequence_len,
                                                             *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        assert encoder is not None, "Encoder must not be None when running sequential DAM"
        self.sequence_len = sequence_len
        self.exhaustive = True
        self.encoder = encoder
        self.dam = DAMLayer(in_channels * modalities, reduction=2)
        self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
        self.att_lstm = SequenceAttentiveLSTM(LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, LATENT_CONV_C * modalities, 3, 3,
                                              sequence_len=sequence_len,
                                              sequence_norm=sequence_norm)
        self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):

        fusion = []
        dam_sals = []
        for seq_idx in range(0, self.sequence_len):
            dam_sal = self.dam(modules[:, seq_idx, ::].clone())
            dam_sals.append(dam_sal)
            modules[:, seq_idx, ::] = self.dam(modules[:, seq_idx, ::].clone(), detached=True)
            mod_fusion=[]
            for mod_idx, mod_step in enumerate(range(0, self.in_channels*self.modalities, self.in_channels)):
                mod_fusion.append(self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step+self.in_channels, ::].clone()))
            mod_fusion = torch.cat(mod_fusion, 1)
            fusion.append(mod_fusion)
        fusion = torch.stack(fusion, 1)
        dam_sals = torch.stack(dam_sals, 1)

        sal = self.att_lstm(fusion)
        sal, lateral = self.gmu(sal)
        sal = self.saliency_out(sal)
        return sal, dam_sals, lateral  # lateral is non-fusion

    
@ModelRegistrar.register
class GASPEncGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", **kwargs):
        super(GASPEncGMUConv, self).__init__(in_channels=in_channels,
                                            modalities=modalities,
                                            out_channels=out_channels,
                                            sequence_len=1,
                                            *args, **kwargs)
        self.encoder = encoder
        if encoder is not None:
            self.enc = ModalityEncoder(in_channels, LATENT_CONV_C, encoder)
            self.gmu = GMUConv2d(LATENT_CONV_C, LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        else:
            self.gmu = GMUConv2d(in_channels, LATENT_CONV_C, modalities, kernel_size=3, padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for mod_idx in range(0, self.in_channels*self.modalities, self.in_channels):
                fusion.append(self.enc(modules[:, mod_idx:mod_idx+self.in_channels, ::]))
            fusion = torch.cat(fusion, 1)
        else:
            fusion = modules  # to operate without an encoder
        sal, lateral = self.gmu(fusion)
        sal = self.saliency_out(sal)
        return sal, None, lateral


@ModelRegistrar.register
class SequenceGASPEncRGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", sequence_len=16, **kwargs):
        super(SequenceGASPEncRGMUConv, self).__init__(in_channels=in_channels,
                                                      modalities=modalities,
                                                      out_channels=out_channels,
                                                      sequence_len=sequence_len,
                                                      *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        self.sequence_len = sequence_len
        self.exhaustive = False
        self.encoder = encoder
        if encoder is not None:
            self.enc_list = nn.ModuleList(
                [ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
            self.gmu = RGMUConv2d(LATENT_CONV_C, LATENT_CONV_C, kernel_size=3, modalities=modalities,
                                  input_size=(self.trg_img_height, self.trg_img_width), padding=1)
        else:
            self.gmu = RGMUConv2d(in_channels, LATENT_CONV_C, kernel_size=3, modalities=modalities,
                                  input_size=(self.inp_img_height, self.inp_img_width), padding=1)

        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        if self.encoder is not None:
            fusion = []
            for seq_idx in range(0, self.sequence_len):
                mod_fusion = []
                for mod_idx, mod_step in enumerate(range(0, self.in_channels * self.modalities, self.in_channels)):
                    mod_fusion.append(
                        self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step + self.in_channels, ::]))
                mod_fusion = torch.cat(mod_fusion, 1)
                fusion.append(mod_fusion)
            fusion = torch.stack(fusion, 1)
        else:
            fusion = modules
        h_l, z_l = self.gmu.initialize_lateral_state()
        lateral = (h_l.to("cuda"), z_l.to("cuda"))
        sal, lateral = self.gmu(fusion, lateral)
        sal = self.saliency_out(sal)
        return sal, None, lateral  # lateral is all


@ModelRegistrar.register
class SequenceGASPDAMEncRGMUConv(GASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, encoder="Conv", sequence_len=16, **kwargs):
        super(SequenceGASPDAMEncRGMUConv, self).__init__(in_channels=in_channels,
                                                      modalities=modalities,
                                                      out_channels=out_channels,
                                                      sequence_len=sequence_len,
                                                      *args, **kwargs)

        # model and dataset mode dependent
        assert sequence_len > 1, "Sequence length must be greater than 1"
        assert encoder is not None, "Encoder must not be None when running sequential DAM"
        self.sequence_len = sequence_len
        self.exhaustive = True
        self.encoder = encoder
        self.dam = DAMLayer(in_channels * modalities, reduction=2)
        self.enc_list = nn.ModuleList(
            [ModalityEncoder(in_channels, LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
        self.gmu = RGMUConv2d(LATENT_CONV_C, LATENT_CONV_C, kernel_size=3, modalities=modalities,
                              input_size=(self.trg_img_height, self.trg_img_width), padding=1)
        self.saliency_out = nn.Conv2d(LATENT_CONV_C, out_channels, kernel_size=1)

    def forward(self, modules):
        fusion = []
        dam_sals = []
        for seq_idx in range(0, self.sequence_len):
            dam_sal = self.dam(modules[:, seq_idx, ::].clone())
            dam_sals.append(dam_sal)
            modules[:, seq_idx, ::] = self.dam(modules[:, seq_idx, ::].clone(), detached=True)
            mod_fusion = []
            for mod_idx, mod_step in enumerate(range(0, self.in_channels * self.modalities, self.in_channels)):
                mod_fusion.append(
                    self.enc_list[mod_idx](modules[:, seq_idx, mod_step:mod_step + self.in_channels, ::].clone()))
            mod_fusion = torch.cat(mod_fusion, 1)
            fusion.append(mod_fusion)
        fusion = torch.stack(fusion, 1)
        dam_sals = torch.stack(dam_sals, 1)

        h_l, z_l = self.gmu.initialize_lateral_state()
        lateral = (h_l.to("cuda"), z_l.to("cuda"))
        sal, lateral = self.gmu(fusion, lateral)
        sal = self.saliency_out(sal)
        return sal, dam_sals, lateral  # lateral is all
