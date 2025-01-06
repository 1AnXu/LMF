"""
Modified from: https://github.com/jaewon-lee-b/lte
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np
from models.arch_ciaosr.arch_csnln import CrossScaleAttention

class ELTE_SAA(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256,                 
                 non_local_attn=True,
                 attn_ensemble=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 **kwargs):
        super().__init__()        
        self.encoder = models.make(encoder_spec)
        imnet_dim = self.encoder.out_dim
        self.feat_dim = self.encoder.out_dim
        # if non_local_attn:
        #     self.feat_dim *= 2
        self.coef = nn.Conv2d(self.feat_dim, hidden_dim, 3, padding=1) # (b,c,h,w)
        self.freq = nn.Conv2d(self.feat_dim, hidden_dim//2, 3, padding=1) # (b,c/2,h,w)
        self.coord_fc = nn.Linear(2, hidden_dim//2, bias=False) #(b*q,2) -> (b*q,c/2)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)        
        
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.weight_attn = AttentionModel(4, 4)
        self.non_local_attn = non_local_attn
        self.attn_ensemble = attn_ensemble
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale
        
        if self.non_local_attn:
            self.non_local_attn_dim = imnet_dim * len(multi_scale)
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)
        
        # For time evaluation
        self.t_total = []
        self.feat_coord = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)

        if self.feat_coord is None or self.feat_coord.shape[-2] != self.feat.shape[-2] \
                or self.feat_coord.shape[-1] != self.feat.shape[-1]:
            self.feat_coord = make_coord(self.feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])

        self.t1 = time.time()

        # 增加非局部注意力
        B, C, H, W = self.feat.shape
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

                        crop_feat = self.feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat_v[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                               pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                               pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat_v = self.cs_attn(self.feat)  # [16, 64, 48, 48]
            self.feat = torch.cat([self.feat, self.non_local_feat_v], dim=1)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        if self.attn_ensemble:
            distances = []
        else:
            areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
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
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                # basis generation
                bs, q = coord.shape[:2]
                # q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)#(b,q,c/2,2)
                # q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))#(b,q,c/2,2) mul (b,q,2,1) -> (b,q,2,c/2)
                # q_freq = torch.sum(q_freq, dim=-2)# (b,q,c/2)
                q_freq *= self.coord_fc(rel_coord.view((bs * q, -1))).view(bs, q, -1)
                
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1) # (b,q,c)

                inp = torch.mul(q_coef, q_freq)

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1) # (b,q,3)
                preds.append(pred)
                if self.attn_ensemble:
                    distance = torch.norm(rel_coord, dim=-1)
                    distances.append(distance)
                else:
                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) # (b,q)
                    areas.append(area + 1e-9)
        if not self.attn_ensemble:
            tot_area = torch.stack(areas).sum(dim=0)
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            
            ret = 0
            for pred, area in zip(preds, areas):
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
        else:
            distances = torch.stack(distances, dim=-1).unsqueeze(-1) # (b,q,4,1)
            preds = torch.stack(preds, dim=-2) # (b,q,4,3)
            ret = self.weight_attn(preds, distances)
        # lr skip
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            if coord is None and cell is None:
                # Evaluate encoder efficiency
                self.feat = self.encoder(inp)
                return None

            self.gen_feat(inp)

            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)

            self.t_total.append(time.time() - self.t1)
            if len(self.t_total) >= 100:
                print(sum(self.t_total[1:]) / (len(self.t_total) - 1))
        return pred

    def forward(self, inp, coord, cell, bsize=None):
        if bsize is not None:
            return self.batched_predict(inp, coord, cell, bsize)

        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        # 定义查询、键和值的线性变换
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        # 用于生成最终RGB预测的全连接层
        self.output_layer = nn.Linear(hidden_dim, 3)  # 输出RGB值（3个通道）

    def forward(self, preds, distances):
        """
        preds: (b, q, 3, 4) -> 每个采样点的4个邻域点的RGB预测值
        distances: (b, q, 4) -> 每个采样点的4个邻域点到目标点的距离
        """
        batch_size, num_samples, _, num_neighbors = preds.size()

        # 1. 合并RGB预测值和距离信息，构造每个邻域点的特征
        # 将RGB和距离拼接在一起，得到(b, q, 4, 4)，即每个邻域点的特征包含RGB和距离
        combined_input = torch.cat([preds, distances.unsqueeze(-2)], dim=-1)  # (b, q, 4, 4)

        # 2. 生成查询 (Q)、键 (K) 和值 (V)
        Q = self.query_linear(combined_input)  # (b, q, 4, hidden_dim)
        K = self.key_linear(combined_input)    # (b, q, 4, hidden_dim)
        V = self.value_linear(combined_input)  # (b, q, 4, hidden_dim)

        # 3. 计算注意力得分（点积）
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))  # (b, q, 4, 4)

        # 4. 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (b, q, 4, 4)

        # 5. 使用注意力权重对值进行加权求和
        attention_output = torch.matmul(attention_weights, V)  # (b, q, 4, hidden_dim)

        # 6. 输出层生成最终RGB值
        output = self.output_layer(attention_output)  # (b, q, 4, 3)

        # 7. 对每个采样点的RGB值进行加权求和，得到目标RGB值
        final_rgb = torch.sum(attention_weights * output, dim=-2)  # (b, q, 3)

        return final_rgb


@register('elte_saa_w_nla')
def make_elte_saa(encoder_spec, imnet_spec=None, hidden_dim=256,                 
                 non_local_attn=True,
                 attn_ensemble=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 **kwargs):
    return ELTE_SAA(encoder_spec, imnet_spec, hidden_dim, non_local_attn, attn_ensemble, multi_scale, softmax_scale)

@register('elte_saa_wo_nla')
def make_elte_saa(encoder_spec, imnet_spec=None, hidden_dim=256,                 
                 non_local_attn=False,
                 attn_ensemble=False,
                 multi_scale=[2],
                 softmax_scale=1,
                 **kwargs):
    return ELTE_SAA(encoder_spec, imnet_spec, hidden_dim, non_local_attn, attn_ensemble, multi_scale, softmax_scale)

@register('elte_saa_wo_nla_w_ae')
def make_elte_saa(encoder_spec, imnet_spec=None, hidden_dim=256,                 
                 non_local_attn=False,
                 attn_ensemble=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 **kwargs):
    return ELTE_SAA(encoder_spec, imnet_spec, hidden_dim, non_local_attn, attn_ensemble, multi_scale, softmax_scale)