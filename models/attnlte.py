"""
Modified from https://github.com/caojiezhang/CiaoSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from models.arch_ciaosr.arch_csnln import CrossScaleAttention
import numpy as np

@register('attnlte')
class ATTNLTE(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 local_size=2,
                 feat_unfold=False,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 hidden_dim=256,
                 **kwargs
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale

        # imnet
        self.encoder = models.make(encoder, args={'no_upsampling': True})
        
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)       
        
        
        imnet_dim = hidden_dim # self.encoder.embed_dim if hasattr(self.encoder, 'embed_dim') else self.encoder.out_dim
        if self.feat_unfold:
            imnet_q_in_dim = imnet_dim * 9
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim * 9
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim * 9
        else:
            imnet_q_in_dim= imnet_dim
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim

        imnet_k_in_dim += 4
        imnet_v_in_dim += 4

        if self.non_local_attn:
            imnet_q_in_dim += imnet_dim * len(multi_scale)
            imnet_v_in_dim += imnet_dim * len(multi_scale)
            imnet_v_out_dim += imnet_dim * len(multi_scale)

        self.imnet_q = models.make(imnet_q, args={'in_dim': imnet_q_in_dim})
        self.imnet_k = models.make(imnet_k, args={'in_dim': imnet_k_in_dim, 'out_dim': imnet_k_out_dim})
        self.imnet_v = models.make(imnet_v, args={'in_dim': imnet_v_in_dim, 'out_dim': imnet_v_out_dim})

        if self.non_local_attn:
            self.non_local_attn_dim = imnet_dim * len(multi_scale)
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

        self.feat_coord = None

    def gen_feat(self, inp):
        self.inp = inp
        feat = self.encoder(inp)
        '''
        if hasattr(self.encoder, 'embed_dim'):
            # SwinIR
            feat = self.encoder.check_image_size(inp)
            feat = self.encoder.conv_first(feat)
            feat = self.encoder.conv_after_body(self.encoder.forward_features(feat)) + feat
        else:
            feat = self.encoder(inp)
        '''

        if self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2] \
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        B, C, H, W = feat.shape
        if self.non_local_attn:
            crop_h, crop_w = 48, 48
            if H * W > crop_h * crop_w:
                # Fixme: generate cross attention by image patches
                self.non_local_feat_v = torch.zeros(B, self.non_local_attn_dim, H, W).cuda()
                for i in range(H // crop_h):
                    for j in range(W // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < H // crop_h - 1 else H)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < W // crop_w - 1 else W)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (
                            padding if i2 + padding <= H else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (
                            padding if j2 + padding <= W else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat_v[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                               pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                               pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat_v = self.cs_attn(feat)  # [16, 64, 48, 48]

        self.coeff = self.coef(feat)
        self.freqq = self.freq(feat)
        self.feat = feat
        return self.feat

    def query_rgb(self, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """
        coef = self.coeff
        freq = self.freqq
        bs, h, w = coef.shape[0], coef.shape[-2], coef.shape[-1]
        coef = coef.permute(0, 2, 3, 1).contiguous().view(bs, -1, coef.shape[1])
        freq = freq.permute(0, 2, 3, 1).contiguous().view(bs, -1, freq.shape[1]) #(b,h*w,c)

        freq = torch.stack(torch.split(freq, 2, dim=-1), dim=-1) # 【c/2 *(b,q,2)】->【(b,q,2,c/2)】
        # freq = torch.mul(freq, rel_coord.unsqueeze(-1))
        freq = torch.sum(freq, dim=-2) # (b,q,c/2)
        if self.cell_decode:
            # Use relative height, width info
            rel_cell = scale.clone()[:, :h * w, :]
            rel_cell[:, :, 0] *= h
            rel_cell[:, :, 1] *= w
            freq += self.phase(rel_cell.view((bs * h * w, -1))).view(bs, h * w, -1)
        freq = torch.cat((torch.cos(np.pi * freq), torch.sin(np.pi * freq)), dim=-1)

        feature = torch.mul(coef, freq).contiguous().view(bs * h * w, -1)  # [16*2304, 640]
        
        B, C, H, W = feature.shape  # [16, 64, 48, 48]
        # query
        query = F.grid_sample(feature, coord.flip(-1).unsqueeze(1), mode='nearest',
                                align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 576]

        feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])  # [16, 2, 48, 48]
        feat_coord = feat_coord.to(coord)
        feat_coord = self.feat_coord

        if self.local_size == 1:
            v_lst = [(0, 0)]
        else:
            v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]
        eps_shift = 1e-6
        preds_k, preds_v = [], []

        for v in v_lst:
            vx, vy = v[0], v[1]
            # project to LR field
            tx = ((H - 1) / (1 - scale[:, 0, 0])).view(B, 1)  # [16, 1]
            ty = ((W - 1) / (1 - scale[:, 0, 1])).view(B, 1)  # [16, 1]
            rx = (2 * abs(vx) - 1) / tx if vx != 0 else 0  # [16, 1]
            ry = (2 * abs(vy) - 1) / ty if vy != 0 else 0  # [16, 1]

            bs, q = coord.shape[:2]
            coord_ = coord.clone()  # [16, 2304, 2]
            if vx != 0:
                coord_[:, :, 0] += vx / abs(vx) * rx + eps_shift  # [16, 2304]
            if vy != 0:
                coord_[:, :, 1] += vy / abs(vy) * ry + eps_shift  # [16, 2304]
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_coef = F.grid_sample(
                coef, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_freq = F.grid_sample(
                freq, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feature.shape[-2]
            rel_coord[:, :, 1] *= feature.shape[-1]
            
            # prepare cell
            rel_cell = scale.clone()
            rel_cell[:, :, 0] *= feature.shape[-2]
            rel_cell[:, :, 1] *= feature.shape[-1]
            # basis generation
            bs, q = coord.shape[:2]
            q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)#(b,q,c/2,2)
            q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))#(b,q,c/2,2) mul (b,q,2,1) -> (b,q,2,c/2)
            q_freq = torch.sum(q_freq, dim=-2)# (b,q,c/2)
            q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
            q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1) # (b,q,c)

            inp = torch.mul(q_coef, q_freq).contiguous().view(bs * q, -1)
            
            feat_k = feat_v = inp
            # key and value
            key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]
            value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 640]

            inp_k = torch.cat([key,rel_coord, rel_cell], dim=-1).contiguous().view(bs * q, -1)  # [16, 2304, 580]
            inp_v = torch.cat([value,rel_coord, rel_cell], dim=-1).contiguous().view(bs * q, -1)  # [16, 2304, 644]

            weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()  # [16, 2304, 576]
            pred_k = (key * weight_k).view(bs, q, -1)  # [16, 2304, 576]

            weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()  # [16, 2304, 576]
            pred_v = (value * weight_v).view(bs, q, -1)  # [16, 2304, 576]

            preds_v.append(pred_v)
            preds_k.append(pred_k)

        preds_k = torch.stack(preds_k, dim=-1)  # [16, 2304, 576, 4]
        preds_v = torch.stack(preds_v, dim=-2)  # [16, 2304, 4, 640]

        attn = (query @ preds_k)  # [16, 2304, 1, 4]
        x = ((attn / self.softmax_scale).softmax(dim=-1) @ preds_v)  # [16, 2304, 1, 640]
        x = x.view(bs * q, -1)  # [16*2304, 640]

        result = self.imnet_q(x)  # [16, 2304, 3]
        result = result.view(bs, q, -1)

        result += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                                padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return result

    def batched_predict(self, x, coord, cell, eval_bsize):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            if coord is None and cell is None:
                # Evaluate encoder efficiency
                feat = self.encoder(x)
                return None

            self.gen_feat(x)
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + eval_bsize, n)
                pred = self.query_rgb(coord[:, left:right, :], cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self, x, coord, cell, bsize=None):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        if bsize is not None:
            pred = self.batched_predict(x, coord, cell, bsize)
        else:
            self.gen_feat(x)
            pred = self.query_rgb(coord, cell)

        return pred