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

class MLPMultiHeadAttention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=1):
        super(MLPMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 输入投影
        self.proj_in = nn.Linear(dim, hidden_dim)
        
        # MLP用于计算注意力
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads * self.head_dim)
        )
        
        # 输出投影
        self.proj_out = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        
        # 输入投影
        h = self.proj_in(x)  # (B, N, hidden_dim)
        
        # MLP计算注意力
        attention_weights = F.softmax(self.mlp(h), dim=1)  # (B, N, num_heads * head_dim)
        attention_weights = attention_weights.view(B, N, self.num_heads, self.head_dim)  # 多头
        
        # 注意力加权
        weighted_x = torch.einsum('bnhd,bnd->bnd', attention_weights, x)  # (B, N, hidden_dim)
        
        # 输出投影
        out = self.proj_out(weighted_x)  # (B, N, D)
        
        return out

@register('lte')
class LTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256,**kwargs):
        super().__init__()        
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)        

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

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
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)#(b,q,c/2,2)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))#(b,q,c/2,2) mul (b,q,2,1) -> (b,q,2,c/2)
                q_freq = torch.sum(q_freq, dim=-2)# (b,q,c/2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1) # (b,q,c)

                inp = torch.mul(q_coef, q_freq)

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
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
