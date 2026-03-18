"""
Hybrid Adaptive Vision Transformer — main_adaptive.py
=====================================================
Architecture:
  - Independent Phase : num_indep_layers 개의 고유 가중치 Transformer Block
  - Shared Phase      : 1개의 Block을 num_shared_recurrences 번 반복 (Recurrent)
  - Decoupled Tokens  : [CLS](idx 0) + [LOSS](idx 1) 을 시퀀스 앞에 Concat
  - Depth Embedding   : Shared Phase 루프마다 [LOSS] 토큰에만 깊이 임베딩을 더함
  - Outputs           : Shared Phase 매 반복의 (cls_out, loss_out)을 수집·반환

Usage:
    python main_adaptive.py \
        --data_path /path/to/imagefolder \
        --num_indep_layers 6 --num_shared_recurrences 6 --epochs 100
"""
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from functools import partial
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import utils
from utils import trunc_normal_
from vision_transformer import Block, PatchEmbed, DINOHead

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Hybrid Adaptive Vision Transformer
# ═════════════════════════════════════════════════════════════════════════════
class HybridAdaptiveVisionTransformer(nn.Module):
    """
    전반부: Independent Phase  (각각 고유 가중치를 갖는 Block × num_indep_layers)
    후반부: Shared Phase       (단 1개 Block 인스턴스 × num_shared_recurrences 반복)

    Decoupled tokens:
        idx 0 → [CLS]  : DINO 증류 손실용 의미론적 특징
        idx 1 → [LOSS] : Early-Exit / Efficiency Pressure / 깊이 인지

    Depth embedding:
        Shared Phase 루프마다 [LOSS] 토큰에만 depth_embeds[l] 을 더한다.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=None,
        num_indep_layers: int = 6,
        num_shared_recurrences: int = 6,
        **kwargs,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_indep_layers = num_indep_layers
        self.num_shared_recurrences = num_shared_recurrences

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches          # e.g. 196

        # ── Decoupled Special Tokens ──
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))   # [CLS]
        self.loss_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # [LOSS]
        # pos_embed 크기: 1 × (num_patches + 2) × embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Independent Phase: 고유 가중치 Block 리스트 ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_indep_layers + 1)]
        self.indep_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(num_indep_layers)
        ])

        # ── Shared Phase: 단일 Block 인스턴스 (반복 사용) ──
        self.shared_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[-1],
            norm_layer=norm_layer,
        )

        # ── Shared Phase 출력 정규화 (깊이별 분포가 다르므로 공유하지 않음) ──
        self.shared_norms = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(num_shared_recurrences)
        ])

        # ── Depth Positional Embedding: [LOSS] 토큰 전용 ──
        # shape: [num_shared_recurrences, embed_dim]
        self.depth_embeds = nn.Parameter(
            torch.zeros(num_shared_recurrences, embed_dim)
        )

        # ── 호환용 Identity Head ──
        self.head = nn.Identity()

        # ── 가중치 초기화 ──
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.loss_token, std=0.02)
        trunc_normal_(self.depth_embeds, std=0.02)
        self.apply(self._init_weights)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # --------------------------------------------------------------------- #
    def interpolate_pos_encoding(self, x, w, h):
        """입력 해상도가 학습 시와 다를 때 pos_embed 보간."""
        npatch = x.shape[1] - 2                          # special 토큰 2개 제외
        N = self.pos_embed.shape[1] - 2
        if npatch == N and w == h:
            return self.pos_embed
        # special 토큰 2개 분리
        special_pos = self.pos_embed[:, :2]               # [1, 2, D]
        patch_pos   = self.pos_embed[:, 2:]               # [1, N, D]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos = nn.functional.interpolate(
            patch_pos.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos.shape[-2] and int(h0) == patch_pos.shape[-1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((special_pos, patch_pos), dim=1)   # [1, 2+npatch, D]

    # --------------------------------------------------------------------- #
    def prepare_tokens(self, x):
        """
        패치 임베딩 → [CLS], [LOSS] Concat → 위치 인코딩
        Returns: [B, 2 + num_patches, embed_dim]
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)                            # [B, num_patches, D]
        cls_tokens  = self.cls_token.expand(B, -1, -1)     # [B, 1, D]
        loss_tokens = self.loss_token.expand(B, -1, -1)    # [B, 1, D]
        x = torch.cat((cls_tokens, loss_tokens, x), dim=1) # [B, 2+P, D]
        x = x + self.interpolate_pos_encoding(x, w, h)    # [B, 2+P, D]
        return self.pos_drop(x)

    # --------------------------------------------------------------------- #
    def forward(self, x):
        """
        Returns:
            cls_intermediates  : list of [B, embed_dim]  — Shared Phase 매 반복의 [CLS] 출력
            loss_intermediates : list of [B, embed_dim]  — Shared Phase 매 반복의 [LOSS] 출력
        """
        x = self.prepare_tokens(x)                         # [B, 2+P, D]

        # ── Independent Phase ──
        for blk in self.indep_blocks:
            x = blk(x)                                     # [B, 2+P, D]

        # ── Shared (Recurrent) Phase ──
        cls_intermediates  = []
        loss_intermediates = []
        for layer_idx in range(self.num_shared_recurrences):
            # In-place 수정을 피하기 위해 새로운 텐서를 생성하여 Re-assemble
            cls_t = x[:, 0:1, :]
            loss_t = x[:, 1:2, :] + self.depth_embeds[layer_idx].view(1, 1, -1)
            patch_t = x[:, 2:, :]
            x = torch.cat([cls_t, loss_t, patch_t], dim=1)

            x = self.shared_block(x)                        # [B, 2+P, D]

            normed = self.shared_norms[layer_idx](x)        # [B, 2+P, D]
            cls_out  = normed[:, 0]                         # [B, D]
            loss_out = normed[:, 1]                         # [B, D]

            cls_intermediates.append(cls_out)
            loss_intermediates.append(loss_out)

        return cls_intermediates, loss_intermediates

    # --------------------------------------------------------------------- #
    def forward_inference(self, images, loss_predictor, heads, exit_prob=0.5):
        """
        인퍼런스 시 Early Exit: [LOSS] 토큰의 예측 확률이 exit_prob 이상이면 종료.
        Returns:
            logits     : [B, out_dim]
            exit_layer : int (0-indexed within shared phase)
        """
        x = self.prepare_tokens(images)                     # [B, 2+P, D]

        # Independent Phase (변경 없이 통과)
        for blk in self.indep_blocks:
            x = blk(x)

        # Shared Phase with Early Exit
        for layer_idx in range(self.num_shared_recurrences):
            cls_t = x[:, 0:1, :]
            loss_t = x[:, 1:2, :] + self.depth_embeds[layer_idx].view(1, 1, -1)
            patch_t = x[:, 2:, :]
            x = torch.cat([cls_t, loss_t, patch_t], dim=1)

            x = self.shared_block(x)
            normed = self.shared_norms[layer_idx](x)
            cls_out  = normed[:, 0]                         # [B, D]
            loss_out = normed[:, 1]                         # [B, D]

            with torch.no_grad():
                pred_logit = loss_predictor(loss_out, layer_idx)  # [B]
                pred_prob  = torch.sigmoid(pred_logit)       # [B]

            if torch.all(pred_prob >= exit_prob):
                logits = heads[layer_idx](cls_out)
                return logits, layer_idx

        logits = heads[-1](cls_out)
        return logits, self.num_shared_recurrences - 1


# ═════════════════════════════════════════════════════════════════════════════
# 2. Loss Predictor — [LOSS] 토큰 feature → early-exit 가능성 logit
# ═════════════════════════════════════════════════════════════════════════════
class LossPredictor(nn.Module):
    """
    Shared Phase 각 반복의 [LOSS] 토큰 feature → BCE logit.
    레이어마다 독립 MLP를 사용한다.
    """
    def __init__(self, embed_dim: int, num_layers: int, hidden_dim: int = 128):
        super().__init__()
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),   # logit (no activation)
            )
            for _ in range(num_layers)
        ])

    def forward(self, z, layer_idx):
        """
        Args:
            z         : [B, embed_dim]  — [LOSS] 토큰 feature
            layer_idx : int             — shared phase 내 반복 인덱스
        Returns:
            pred_logit: [B]
        """
        return self.predictors[layer_idx](z).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Adaptive DINO Loss (L_distill + L_align + L_pressure)
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, num_layers=6,
                 warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0, nepochs=100,
                 student_temp=0.1, center_momentum=0.9,
                 alpha_align=0.1, alpha_pressure=0.1,
                 align_warmup_init=0.0,
                 align_warmup_start_epoch=10,
                 align_warmup_end_epoch=60,
                 align_warmup_mode="cosine",
                 ema_momentum=0.99, tau_percentile=0.5, bce_pos_weight=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.num_layers = num_layers

        self.alpha_align_target = alpha_align
        self.alpha_pressure = alpha_pressure
        self.align_warmup_init = align_warmup_init
        self.align_warmup_start_epoch = align_warmup_start_epoch
        self.align_warmup_end_epoch = align_warmup_end_epoch
        self.align_warmup_mode = align_warmup_mode

        self.ema_momentum = ema_momentum
        self.tau_percentile = tau_percentile
        
        self.register_buffer("loss_ema", torch.ones(num_layers))
        self.register_buffer("_ema_initialized", torch.tensor(False))

        self.register_buffer("bce_pos_weight_tensor",
                             torch.tensor(float(bce_pos_weight), dtype=torch.float32))
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def get_alpha_align(self, epoch: int) -> float:
        s, e = self.align_warmup_start_epoch, self.align_warmup_end_epoch
        init, target = self.align_warmup_init, self.alpha_align_target
        if e <= s:
            return target if epoch >= s else init
        if epoch < s:
            return init
        if epoch >= e:
            return target
        t = (epoch - s) / float(e - s)
        w = t if self.align_warmup_mode == "linear" else 0.5 * (1.0 - math.cos(math.pi * t))
        return init + (target - init) * w

    def _update_ema_and_get_targets(self, per_layer_losses, epoch):
        """
        레이어별 EMA 업데이트 후 동적 threshold 계산.
        첫 번째 호출 시 EMA를 현재 값으로 초기화 (cold start 방지).
        """
        loss_device = per_layer_losses[0].device
        loss_vals = torch.tensor(
            [per_layer_losses[layer_idx].detach().item() for layer_idx in range(self.num_layers)],
            device=self.loss_ema.device
        )
        
        if not self._ema_initialized.item():
            self.loss_ema.copy_(loss_vals)
            self._ema_initialized.fill_(True)
        else:
            self.loss_ema = (self.ema_momentum * self.loss_ema 
                            + (1 - self.ema_momentum) * loss_vals)
        
        # align_warmup과 연동하여 동적 percentile 계산
        alpha_align_curr = self.get_alpha_align(epoch)
        effective_percentile = self.tau_percentile * (
            alpha_align_curr / self.alpha_align_target
        ) if self.alpha_align_target > 0 else 0.0
        
        # 동적 threshold: 전체 레이어 EMA 중 하위 effective_percentile
        threshold = torch.quantile(self.loss_ema, effective_percentile)
        
        targets = (loss_vals <= threshold).float()  # [num_layers]
        return targets.to(loss_device), threshold.item()

    def forward(self, student_logits_per_layer, teacher_logits,
                loss_predictor, loss_intermediates, epoch):
        """
        Args:
            student_logits_per_layer : list[Tensor]  — 각 shared 반복의 head 출력
            teacher_logits           : Tensor         — teacher 마지막 층 head 출력
            loss_predictor           : LossPredictor
            loss_intermediates       : list[Tensor]   — [LOSS] 토큰 features
            epoch                    : int
        """
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_logits - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # ① L_distill
        per_layer_losses = []
        for layer_idx in range(self.num_layers):
            s_out = student_logits_per_layer[layer_idx] / self.student_temp
            s_out = s_out.chunk(self.ncrops)
            layer_loss, n_terms = 0.0, 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(s_out)):
                    if v == iq:
                        continue
                    loss = torch.sum(-q * F.log_softmax(s_out[v], dim=-1), dim=-1)
                    layer_loss += loss.mean()
                    n_terms += 1
            per_layer_losses.append(layer_loss / n_terms)

        L_distill = sum(per_layer_losses)

        # ── 동적 target 생성 ──
        layer_targets, dynamic_threshold = self._update_ema_and_get_targets(per_layer_losses, epoch)

        # ② L_align (BCE) — [LOSS] 토큰 사용
        B_global = teacher_logits.shape[0]
        align_terms, pred_logits_det, target_det = [], [], []
        for layer_idx in range(self.num_layers):
            z_l = loss_intermediates[layer_idx][:B_global]                 # [2B, D]
            pred_logit_l = loss_predictor(z_l, layer_idx).view(-1)         # [2B]
            target_l = torch.full_like(pred_logit_l, layer_targets[layer_idx].item())
            bce_l = F.binary_cross_entropy_with_logits(
                pred_logit_l, target_l,
                pos_weight=self.bce_pos_weight_tensor.to(pred_logit_l.device))
            align_terms.append(bce_l)
            pred_logits_det.append(pred_logit_l.detach())
            target_det.append(target_l.detach())

        L_align = torch.stack(align_terms).mean()

        # ③ L_pressure — expected depth 최소화
        prob_layers = []
        for layer_idx in range(self.num_layers):
            z_l = loss_intermediates[layer_idx][:B_global]
            pred_logit_l = loss_predictor(z_l, layer_idx).view(-1)
            prob_layers.append(torch.sigmoid(pred_logit_l))
        p = torch.stack(prob_layers, dim=1)                                # [2B, L]
        B_p, L = p.shape
        surv = torch.cumprod(
            torch.cat([torch.ones(B_p, 1, device=p.device), 1.0 - p + 1e-6], dim=1),
            dim=1,
        )[:, :-1]
        exit_dist = surv * p
        exit_dist[:, -1] += (1.0 - exit_dist.sum(dim=1))
        depths = torch.arange(1, L + 1, device=p.device, dtype=p.dtype).unsqueeze(0)
        L_pressure = ((exit_dist * depths).sum(dim=1) / L).mean()

        alpha_align_curr = self.get_alpha_align(epoch)
        # alpha_align이 웜업 중일 때 pressure도 가해지지 않도록 동기화 (Reward Hacking 방지)
        alpha_pressure_curr = self.alpha_pressure * (alpha_align_curr / self.alpha_align_target) if self.alpha_align_target > 0 else 0.0
        total_loss = L_distill + alpha_align_curr * L_align + alpha_pressure_curr * L_pressure

        self.update_center(teacher_logits)

        with torch.no_grad():
            pred_all = torch.cat(pred_logits_det)
            tgt_all  = torch.cat(target_det)
            align_acc = ((pred_all >= 0).float() == tgt_all).float().mean()

        loss_dict = {
            "L_distill": L_distill.item(), "L_align": L_align.item(),
            "L_pressure": L_pressure.item(), "total_loss": total_loss.item(),
            "alpha_align_curr": float(alpha_align_curr),
            "align_acc": align_acc.item(),
            "target_pos_rate": layer_targets.mean().item(),
            "dynamic_threshold": dynamic_threshold,
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



# ═════════════════════════════════════════════════════════════════════════════
# 4. Multi-Crop Wrapper (Hybrid 버전)
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveMultiCropWrapper(nn.Module):
    """
    Multi-resolution crop → backbone → 레이어별 [CLS] head 출력 + [LOSS] features.
    """
    def __init__(self, backbone, heads, loss_predictor):
        super().__init__()
        self.backbone = backbone
        self.heads = heads                # nn.ModuleList of DINOHead
        self.loss_predictor = loss_predictor

    def forward(self, x):
        """
        Returns:
            logits_per_layer   : list[Tensor]  — 각 shared 반복의 head([CLS]) 출력
            loss_intermediates : list[Tensor]   — 각 shared 반복의 [LOSS] features
        """
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        num_layers = self.backbone.num_shared_recurrences
        all_cls  = [[] for _ in range(num_layers)]
        all_loss = [[] for _ in range(num_layers)]

        start_idx = 0
        for end_idx in idx_crops:
            batch = torch.cat(x[start_idx:end_idx])
            cls_outs, loss_outs = self.backbone(batch)  # lists of [B_chunk, D]
            for layer_idx in range(num_layers):
                all_cls[layer_idx].append(cls_outs[layer_idx])
                all_loss[layer_idx].append(loss_outs[layer_idx])
            start_idx = end_idx

        cls_intermediates  = [torch.cat(feats, dim=0) for feats in all_cls]
        loss_intermediates = [torch.cat(feats, dim=0) for feats in all_loss]

        logits_per_layer = [self.heads[layer_idx](cls_intermediates[layer_idx]) for layer_idx in range(num_layers)]

        return logits_per_layer, loss_intermediates


# ═════════════════════════════════════════════════════════════════════════════
# 5. Argument Parser
# ═════════════════════════════════════════════════════════════════════════════
def get_args_parser():
    parser = argparse.ArgumentParser('Hybrid SPAE — Adaptive Encoder (Non-DDP)', add_help=False)

    # ── Model ──
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--num_heads', default=6, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--num_indep_layers', default=6, type=int,
        help='Number of independent (unique-weight) transformer blocks.')
    parser.add_argument('--num_shared_recurrences', default=6, type=int,
        help='Number of recurrences of the single shared block.')
    parser.add_argument('--out_dim', default=4096, type=int)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--img_size', default=224, type=int)

    # ── Loss ──
    parser.add_argument('--alpha_align', default=0.1, type=float)
    parser.add_argument('--alpha_pressure', default=0.1, type=float)
    parser.add_argument('--align_warmup_init', default=0.0, type=float)
    parser.add_argument('--align_warmup_start_epoch', default=10, type=int)
    parser.add_argument('--align_warmup_end_epoch', default=60, type=int)
    parser.add_argument('--align_warmup_mode', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--ema_momentum', default=0.99, type=float)
    parser.add_argument('--tau_percentile', default=0.5, type=float)
    parser.add_argument('--bce_pos_weight', default=1.0, type=float)
    parser.add_argument('--early_exit_prob', default=0.5, type=float)
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)

    # ── Training ──
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # ── Multi-crop ──
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))

    # ── Misc ──
    parser.add_argument('--data_path', default='./cifar100_images/train', type=str)
    parser.add_argument('--output_dir', default='./output_adaptive_hybrid', type=str)
    parser.add_argument('--tensorboard_log_dir', default='', type=str,
                        help='TensorBoard log directory. Empty uses <output_dir>/tensorboard.')
    parser.add_argument('--saveckp_freq', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    return parser


# ═════════════════════════════════════════════════════════════════════════════
# 6. Training
# ═════════════════════════════════════════════════════════════════════════════
def train_adaptive(args):
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(vars(args).items())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    tb_log_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else Path(args.output_dir) / "tensorboard"
    tb_writer = None
    if SummaryWriter is not None and utils.is_main_process():
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logs: {tb_log_dir}")
    elif SummaryWriter is None and utils.is_main_process():
        print("TensorBoard is unavailable because torch.utils.tensorboard could not be imported.")

    # ── data ──
    from main_dino import DataAugmentationDINO
    transform = DataAugmentationDINO(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    print(f"Data loaded: {len(dataset)} images.")

    # ── model ──
    ncrops = args.local_crops_number + 2
    embed_dim = args.embed_dim
    num_shared = args.num_shared_recurrences

    def _build_model():
        backbone = HybridAdaptiveVisionTransformer(
            img_size=args.img_size, patch_size=args.patch_size,
            embed_dim=embed_dim, num_heads=args.num_heads,
            drop_path_rate=args.drop_path_rate,
            num_indep_layers=args.num_indep_layers,
            num_shared_recurrences=num_shared)
        heads = nn.ModuleList([
            DINOHead(embed_dim, args.out_dim,
                     use_bn=args.use_bn_in_head,
                     norm_last_layer=args.norm_last_layer if layer_idx == num_shared - 1 else True)
            for layer_idx in range(num_shared)])
        predictor = LossPredictor(embed_dim, num_shared)
        return AdaptiveMultiCropWrapper(backbone, heads, predictor)

    student = _build_model().to(device)
    teacher = _build_model().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Hybrid model built: indep={args.num_indep_layers}, shared={num_shared}, "
          f"embed_dim={embed_dim}, device={device}")

    # ── loss ──
    adaptive_loss = AdaptiveDINOLoss(
        out_dim=args.out_dim, ncrops=ncrops, num_layers=num_shared,
        warmup_teacher_temp=args.warmup_teacher_temp, teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs, nepochs=args.epochs,
        alpha_align=args.alpha_align, alpha_pressure=args.alpha_pressure,
        align_warmup_init=args.align_warmup_init,
        align_warmup_start_epoch=args.align_warmup_start_epoch,
        align_warmup_end_epoch=args.align_warmup_end_epoch,
        align_warmup_mode=args.align_warmup_mode,
        ema_momentum=args.ema_momentum, tau_percentile=args.tau_percentile,
        bce_pos_weight=args.bce_pos_weight,
    ).to(device)

    # ── optimizer ──
    backbone_params = list(student.backbone.parameters())
    head_params = list(student.heads.parameters())
    predictor_params = list(student.loss_predictor.parameters())
    params_groups = [
        {"params": [p for p in backbone_params if p.requires_grad and p.ndim >= 2],
         "weight_decay": args.weight_decay},
        {"params": [p for p in backbone_params if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
        {"params": head_params, "weight_decay": 0.0},
        {"params": predictor_params, "weight_decay": 0.0},
    ]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)

    fp16_scaler = None
    if args.use_fp16 and device.type == "cuda":
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ── schedulers ──
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size_per_gpu / 256., args.min_lr,
        args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader))
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # ── resume ──
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher, optimizer=optimizer,
        fp16_scaler=fp16_scaler, adaptive_loss=adaptive_loss)
    start_epoch = to_restore["epoch"]

    # ── train loop ──
    start_time = time.time()
    print("Starting Hybrid SPAE training!")
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student, teacher, adaptive_loss, data_loader,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, device, tb_writer)

        save_dict = {
            'student': student.state_dict(), 'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
            'args': args, 'adaptive_loss': adaptive_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if tb_writer is not None:
                for key, value in train_stats.items():
                    tb_writer.add_scalar(f"train/{key}", value, epoch)

    if tb_writer is not None:
        tb_writer.close()

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


def train_one_epoch(student, teacher, adaptive_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args, device, tb_writer=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[global_it]

        images = [im.to(device, non_blocking=(device.type == "cuda")) for im in images]

        amp_ctx = torch.cuda.amp.autocast if fp16_scaler is not None else nullcontext
        with amp_ctx():
            # teacher: global views only, [LOSS] features 불필요
            teacher_logits_per_layer, _ = teacher(images[:2])
            teacher_logits = teacher_logits_per_layer[-1]

            # student: 모든 crop
            logits_per_layer, loss_intermediates = student(images)

            loss, loss_dict = adaptive_loss(
                logits_per_layer, teacher_logits,
                student.loss_predictor, loss_intermediates, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update
        with torch.no_grad():
            m = momentum_schedule[global_it]
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(total_loss=loss_dict["total_loss"])
        metric_logger.update(L_distill=loss_dict["L_distill"])
        metric_logger.update(L_align=loss_dict["L_align"])
        metric_logger.update(L_pressure=loss_dict["L_pressure"])
        if "alpha_align_curr" in loss_dict:
            metric_logger.update(alpha_align=loss_dict["alpha_align_curr"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if global_it % 50 == 0 and utils.is_main_process():
            if "align_acc" in loss_dict:
                print(f"  [BCE Predictor] acc={loss_dict['align_acc']:.4f}, "
                      f"target_pos_rate={loss_dict.get('target_pos_rate', -1):.4f}")

        if tb_writer is not None and utils.is_main_process():
            tb_writer.add_scalar("train_step/total_loss", loss_dict["total_loss"], global_it)
            tb_writer.add_scalar("train_step/L_distill", loss_dict["L_distill"], global_it)
            tb_writer.add_scalar("train_step/L_align", loss_dict["L_align"], global_it)
            tb_writer.add_scalar("train_step/L_pressure", loss_dict["L_pressure"], global_it)
            tb_writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_it)
            if "alpha_align_curr" in loss_dict:
                tb_writer.add_scalar("train_step/alpha_align", loss_dict["alpha_align_curr"], global_it)
            if "align_acc" in loss_dict:
                tb_writer.add_scalar("train_step/align_acc", loss_dict["align_acc"], global_it)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def inference_with_early_exit(model, images, threshold=0.5):
    """threshold: sigmoid(logit) >= 값이면 exit."""
    model.eval()
    with torch.no_grad():
        logits, exit_layer = model.backbone.forward_inference(
            images, model.loss_predictor, model.heads, exit_prob=threshold)
    return logits, exit_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hybrid SPAE', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_adaptive(args)