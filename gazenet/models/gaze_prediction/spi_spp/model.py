import random
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics

from gazenet.utils.registrar import *
from gazenet.models.shared_components.attentive_convlstm.model import SequenceAttentiveLSTM
from gazenet.models.shared_components.gmu.model import GMUConv2d
from gazenet.models.saliency_prediction.gasp.model import GASPBase, ModalityEncoder, DAMLayer
import gazenet.models.saliency_prediction.gasp.model as gasp_model

gasp_model.EVAL_VISUALIZATION_PROBABILITY = 0.001
gasp_model.TRAIN_LOG_ON_STEP = True
gasp_model.TRAIN_LOG_ON_EPOCH = False
gasp_model.LATENT_CONV_C = 32


class SPPGASPBase(GASPBase):
    def __init__(self, num_classes, part_loss_weight=0.5, **kwargs):
        super(SPPGASPBase, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.part_loss_weight = part_loss_weight

        # CAREFUL (fabawi): the following loss_weights_named are signed and inverted
        self.loss_weights_named.update(**{"part_nll_loss": 1/self.part_loss_weight})

        self.accuracy_metric = torchmetrics.Accuracy(multiclass=True, num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1(multiclass=True, num_classes=self.num_classes, average="macro")
        if self.eval_extra_metrics:
            self.loss_weights_named.update(**{
                "part_acc": 1,
                "part_f1": 1})

    def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            logits = self.forward(x)

            if len(logits) > 1 and logits[1] is not None:
                # calculate dam loss if model supports it
                if self.exhaustive:
                    dam_logit_tgt = "seq_" + self.train_dataset_properties["gt_mappings"]["gt_salmap"]
                else:
                    dam_logit_tgt = + self.train_dataset_properties["gt_mappings"]["gt_salmap"]
                dam_logits = F.binary_cross_entropy_with_logits(logits[1], y[dam_logit_tgt]) * self.dam_loss_weight
            else:
                dam_logits = torch.tensor(0)
            part_logits = F.nll_loss(logits[3], y["gt_part"])

            sal_logits = logits[0]
            losses = self.loss(sal_logits, y, self.train_dataset_properties)
            losses.update(dam_bce_loss=dam_logits)
            losses.update(part_nll_loss=part_logits)
            if self.eval_extra_metrics:
                losses.update(part_acc=self.accuracy_metric(torch.argmax(logits[3], dim=1), y["gt_part"]),
                              part_f1=self.f1_metric(torch.argmax(logits[3], dim=1), y["gt_part"]))

            logs = {f'train_{k}': (v * self.loss_weights_named[k])/self.batch_size for k, v in losses.items()}
            self.log_dict(logs, on_step=gasp_model.TRAIN_LOG_ON_STEP, on_epoch=gasp_model.TRAIN_LOG_ON_EPOCH)
            return {"loss": losses["loss"] + dam_logits + part_logits}  # "log": logs

    def validation_step(self, val_batch, batch_idx):
        val_loss = 0
        x, y = val_batch
        logits = self.forward(x)
        if random.random() < gasp_model.EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(x, logits, y, self.val_dataset_properties, batch_idx)
        for sample_idx, (sal_logit, part_logit), in enumerate(zip(logits[0], logits[3])):
            gt = {key: torch.unsqueeze(value[sample_idx], 0) for key, value in y.items()}
            sal_logit = torch.unsqueeze(sal_logit, 0)
            part_logit = torch.unsqueeze(part_logit, 0)
            part_idx = gt["gt_part"][0].item() + 1
            losses = self.loss(sal_logit, gt, self.val_dataset_properties,
                               human_infinite=self.eval_human_infinite, extra_metrics=self.eval_extra_metrics)
            if self.eval_extra_metrics:
                losses.update(**{f"part_acc": self.accuracy_metric(torch.argmax(part_logit, dim=1), gt["gt_part"]),
                                 f"part_f1": self.f1_metric(torch.argmax(part_logit, dim=1), gt["gt_part"])})

            logs = {f"val_{k}_p{part_idx}": (v * self.loss_weights_named.get(k, 1)) for k, v in losses.items()}
            val_loss += losses["loss"]
            self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        val_loss /= self.batch_size
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        return {"val_loss": val_loss}

    def test_step(self, test_batch, batch_idx):
        test_loss = 0
        x, y = test_batch
        logits = self.forward(x)
        if random.random() < gasp_model.EVAL_VISUALIZATION_PROBABILITY and self.val_store_image_samples:
            self.log_val_images(x, logits, y, self.val_dataset_properties, step=batch_idx)
        for sample_idx, (sal_logit, part_logit), in enumerate(zip(logits[0], logits[3])):
            gt = {key: torch.unsqueeze(value[sample_idx], 0) for key, value in y.items()}
            sal_logit = torch.unsqueeze(sal_logit, 0)
            part_logit = torch.unsqueeze(part_logit, 0)
            part_idx = gt["gt_part"][0].item() + 1
            losses = self.loss(sal_logit, gt, self.val_dataset_properties,
                               human_infinite=self.eval_human_infinite, extra_metrics=self.eval_extra_metrics)
            if self.eval_extra_metrics:
                losses.update(**{f"part_acc": self.accuracy_metric(torch.argmax(part_logit, dim=1), gt["gt_part"]),
                                 f"part_f1": self.f1_metric(torch.argmax(part_logit, dim=1), gt["gt_part"])})

            logs = {f"test_{k}_p{part_idx}": (v * self.loss_weights_named.get(k, 1)) for k, v in losses.items()}
            test_loss += losses["loss"]
            self.log_dict(logs, on_step=True, on_epoch=True)
        # return logs
        test_loss /= self.batch_size
        self.log("test_loss", test_loss, on_step=True, on_epoch=True)
        return {"test_loss": test_loss}


@ModelRegistrar.register
class SequenceSPPGASPDAMEncGMUALSTMConv(SPPGASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceSPPGASPDAMEncGMUALSTMConv, self).__init__(in_channels=in_channels,
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
        self.enc_list = nn.ModuleList([ModalityEncoder(in_channels, gasp_model.LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
        self.gmu = GMUConv2d(gasp_model.LATENT_CONV_C, gasp_model.LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.att_lstm = SequenceAttentiveLSTM(gasp_model.LATENT_CONV_C, gasp_model.LATENT_CONV_C, gasp_model.LATENT_CONV_C, 3, 3,
                                              sequence_len=sequence_len,
                                              sequence_norm=sequence_norm)
        self.saliency_out = nn.Conv2d(gasp_model.LATENT_CONV_C, out_channels, kernel_size=1)

        n_sizes = self._get_conv_output((sequence_len, in_channels * modalities, self.inp_img_width, self.inp_img_height))
        self.part_fc1 = nn.Linear(n_sizes, 512)
        self.part_fc2 = nn.Linear(512, 128)
        self.part_out = nn.Linear(128, self.num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)[1]
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, modules):
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
            mod_fusion, lateral = self.gmu(mod_fusion)
            fusion.append(mod_fusion)
        fusion = torch.stack(fusion, 1)
        sal = self.att_lstm(fusion)
        sal = self.saliency_out(sal)
        dam_sals = torch.stack(dam_sals, 1)
        return fusion, sal, dam_sals, lateral

    def forward(self, modules):
        _, sal_out, dam_sals_out, lateral_out = self._forward_features(modules)
        sal_out = F.relu(sal_out, inplace=True)

        part = sal_out.view(sal_out.size(0), -1)
        part = F.relu(self.part_fc1(part))
        part = F.relu(self.part_fc2(part))
        part_out = F.log_softmax(self.part_out(part), dim=1)

        return sal_out, dam_sals_out, lateral_out, part_out  # lateral is only the last


@ModelRegistrar.register
class SequenceSPPGASPDAMEncALSTMGMUConv(SPPGASPBase):
    def __init__(self, *args, in_channels=3, modalities=4, out_channels=1, sequence_len=16, sequence_norm=False, encoder="Conv", **kwargs):
        super(SequenceSPPGASPDAMEncALSTMGMUConv, self).__init__(in_channels=in_channels,
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
            [ModalityEncoder(in_channels, gasp_model.LATENT_CONV_C, encoder=encoder) for _ in range(modalities)])
        self.att_lstm = SequenceAttentiveLSTM(gasp_model.LATENT_CONV_C * modalities, gasp_model.LATENT_CONV_C * modalities,
                                              gasp_model.LATENT_CONV_C * modalities, 3, 3,
                                              sequence_len=sequence_len,
                                              sequence_norm=sequence_norm)
        self.gmu = GMUConv2d(gasp_model.LATENT_CONV_C, gasp_model.LATENT_CONV_C, modalities, kernel_size=3, padding=1)
        self.saliency_out = nn.Conv2d(gasp_model.LATENT_CONV_C, out_channels, kernel_size=1)

        n_sizes = self._get_conv_output((sequence_len, in_channels * modalities, self.inp_img_width, self.inp_img_height))
        self.part_fc1 = nn.Linear(n_sizes, 512)
        self.part_fc2 = nn.Linear(512, 128)
        self.part_out = nn.Linear(128, self.num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)[1]
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, modules):
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
        sal = self.att_lstm(fusion)
        sal, lateral = self.gmu(sal)
        sal = self.saliency_out(sal)
        dam_sals = torch.stack(dam_sals, 1)
        return fusion, sal, dam_sals, lateral

    def forward(self, modules):
        _, sal_out, dam_sals_out, lateral_out = self._forward_features(modules)
        sal_out = F.relu(sal_out, inplace=True)

        part = sal_out.view(sal_out.size(0), -1)
        part = F.relu(self.part_fc1(part))
        part = F.relu(self.part_fc2(part))
        part_out = F.log_softmax(self.part_out(part), dim=1)

        return sal_out, dam_sals_out, lateral_out, part_out  # lateral is only the last
