"""
Hybrid Adaptive Vision Transformer — main_adaptive.py
=====================================================
Architecture:
    - Independent Phase : 4개의 고유 가중치 Transformer Block
    - Shared Phase      : 2개의 Block 세트를 num_shared_recurrences 번 반복 (Recurrent)
    - Decoupled Tokens  : [CLS](idx 0) + [LOSS](idx 1) 을 시퀀스 앞에 Concat
    - Depth Embedding   : Shared Phase 각 반복 시작 시 모든 토큰에 step embedding을 주입
    - Outputs           : Shared Phase 매 반복의 (cls_out, loss_out)을 수집·반환
    - DDP               : torchrun / distributed training 지원

Usage:
        torchrun --standalone --nproc_per_node=2 main_adaptive.py \
                --data_path /path/to/imagefolder \
        --num_indep_layers 4 --num_shared_recurrences 4 --epochs 100
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
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from utils import trunc_normal_
from vision_transformer import Block, PatchEmbed, DINOHead


# ═════════════════════════════════════════════════════════════════════════════
# 1. Hybrid Adaptive Vision Transformer
# ═════════════════════════════════════════════════════════════════════════════
class HybridAdaptiveVisionTransformer(nn.Module):
    """
    전반부: Independent Phase  (4개의 고유 가중치를 갖는 Block)
    후반부: Shared Phase       (2개 Block 인스턴스 × num_shared_recurrences 반복)

    Decoupled tokens:
        idx 0 → [CLS]  : DINO 증류 손실용 의미론적 특징
        idx 1 → [LOSS] : Early-Exit / Efficiency Pressure / 깊이 인지

    Depth embedding:
        Shared Phase 각 반복 시작과 종료에 recurrence-specific step embedding을
        모든 토큰에 broadcast하여 더한다.
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
        num_indep_layers: int = 4,
        num_shared_recurrences: int = 4,
        **kwargs,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_indep_layers = 4
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

        # ── Independent Phase: 고유 가중치 Block 리스트 (4개 고정) ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]
        self.indep_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(self.num_indep_layers)
        ])

        # ── Shared Phase: 2개 Block 인스턴스로 구성된 재사용 세트 ──
        self.shared_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[4 + i],
                norm_layer=norm_layer,
            )
            for i in range(2)
        ])

        # ── Shared Phase 출력 정규화 (깊이별 분포가 다르므로 공유하지 않음) ──
        self.shared_norms = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(num_shared_recurrences)
        ])

        # ── Global Depth Positional Embedding ──
        # Broadcast across the full token sequence so [CLS], [LOSS], and patches
        # all receive the same recurrence-specific step signal.
        self.global_depth_embeds = nn.Parameter(
            torch.zeros(self.num_shared_recurrences, 1, 1, embed_dim)
        )

        # ── 호환용 Identity Head ──
        self.head = nn.Identity()

        # ── 가중치 초기화 ──
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.loss_token, std=0.02)
        trunc_normal_(self.global_depth_embeds, std=0.02)
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
        for l in range(self.num_shared_recurrences):
            step_embed = self.global_depth_embeds[l]
            x = x + step_embed
            for shared_blk in self.shared_blocks:
                x = shared_blk(x)              # [B, 2+P, D]
            normed = self.shared_norms[l](x)

            cls_out  = normed[:, 0]
            loss_out = normed[:, 1]

            cls_intermediates.append(cls_out)
            loss_intermediates.append(loss_out)

        return cls_intermediates, loss_intermediates

    # --------------------------------------------------------------------- #
    def forward_inference(self, images, loss_predictor, head, exit_prob=0.5):
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
        for l in range(self.num_shared_recurrences):
            step_embed = self.global_depth_embeds[l]
            x = x + step_embed
            for shared_blk in self.shared_blocks:
                x = shared_blk(x)              # [B, 2+P, D]
            normed = self.shared_norms[l](x)

            cls_out  = normed[:, 0]
            loss_out = normed[:, 1]

            with torch.no_grad():
                pred_logit = loss_predictor(loss_out, l)     # [B]
                pred_prob  = torch.sigmoid(pred_logit)       # [B]

            if torch.all(pred_prob >= exit_prob):
                logits = head(cls_out)
                return logits, l

        logits = head(cls_out)
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
                 align_warmup_start_epoch=None,
                 align_warmup_end_epoch=None,
                 align_warmup_mode="cosine",
                 pressure_warmup_init=0.0,
                 pressure_warmup_start_epoch=None,
                 pressure_warmup_end_epoch=None,
                 pressure_warmup_mode="cosine",
                 ema_momentum=0.99, tau_percentile=0.5, bce_pos_weight=1.0,
                 distill_mode="last_only"):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.num_layers = num_layers

        self.alpha_align_target = alpha_align
        self.alpha_pressure_target = alpha_pressure
        self.align_warmup_init = align_warmup_init
        self.align_warmup_start_epoch = nepochs // 2 if align_warmup_start_epoch is None else align_warmup_start_epoch
        self.align_warmup_end_epoch = nepochs - 1 if align_warmup_end_epoch is None else align_warmup_end_epoch
        self.align_warmup_mode = align_warmup_mode
        self.pressure_warmup_init = pressure_warmup_init
        self.pressure_warmup_start_epoch = nepochs // 2 if pressure_warmup_start_epoch is None else pressure_warmup_start_epoch
        self.pressure_warmup_end_epoch = nepochs - 1 if pressure_warmup_end_epoch is None else pressure_warmup_end_epoch
        self.pressure_warmup_mode = pressure_warmup_mode

        self.ema_momentum = ema_momentum
        self.tau_percentile = tau_percentile
        self.distill_mode = distill_mode
        
        self.register_buffer("loss_ema", torch.ones(num_layers))
        self.register_buffer("_ema_initialized", torch.tensor(False))

        self.register_buffer("bce_pos_weight_tensor",
                             torch.tensor(float(bce_pos_weight), dtype=torch.float32))
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def _get_distill_weights(self, device):
        if self.distill_mode == "none":
            return torch.zeros(self.num_layers, device=device)
        if self.distill_mode == "last_only":
            weights = torch.zeros(self.num_layers, device=device)
            weights[-1] = 1.0
            return weights
        if self.distill_mode == "linear":
            weights = torch.linspace(0.1, 1.0, self.num_layers, device=device)
            return weights / weights.sum()
        raise ValueError(f"Unknown distill_mode: {self.distill_mode}")

    def _warmup_value(self, epoch: int, start_epoch: int, end_epoch: int,
                      init_value: float, target_value: float, mode: str) -> float:
        if end_epoch <= start_epoch:
            return target_value if epoch >= start_epoch else init_value
        if epoch < start_epoch:
            return init_value
        if epoch >= end_epoch:
            return target_value
        t = (epoch - start_epoch) / float(end_epoch - start_epoch)
        w = t if mode == "linear" else 0.5 * (1.0 - math.cos(math.pi * t))
        return init_value + (target_value - init_value) * w

    def get_alpha_align(self, epoch: int) -> float:
        return self._warmup_value(
            epoch,
            self.align_warmup_start_epoch,
            self.align_warmup_end_epoch,
            self.align_warmup_init,
            self.alpha_align_target,
            self.align_warmup_mode,
        )

    def get_alpha_pressure(self, epoch: int) -> float:
        return self._warmup_value(
            epoch,
            self.pressure_warmup_start_epoch,
            self.pressure_warmup_end_epoch,
            self.pressure_warmup_init,
            self.alpha_pressure_target,
            self.pressure_warmup_mode,
        )

    def _update_ema_and_get_targets(self, inst_losses, epoch):
        """
        inst_losses: [num_layers, B]
        """
        loss_device = inst_losses.device
        loss_vals_mean = inst_losses.mean(dim=1).detach()

        if utils.get_world_size() > 1:
            dist.all_reduce(loss_vals_mean)
            loss_vals_mean /= utils.get_world_size()
        
        self.loss_ema = self.loss_ema.to(loss_device)
        
        if not self._ema_initialized.item():
            self.loss_ema.copy_(loss_vals_mean)
            self._ema_initialized.fill_(1)
        else:
            self.loss_ema = (self.ema_momentum * self.loss_ema 
                            + (1 - self.ema_momentum) * loss_vals_mean)
        
        targets = []
        thresholds = []
        
        for l in range(self.num_layers):
            layer_median = self.loss_ema[l]  # 해당 레이어의 기대 손실
            # 레이어별로 독립적으로 "이 샘플이 이 레이어에서 충분히 수렴했는가" 판단
            target_l = (inst_losses[l].detach() <= layer_median).float()  # [B]
            targets.append(target_l)
            thresholds.append(layer_median.item())
            
        targets = torch.stack(targets)  # [num_layers, B]
        return targets.to(loss_device), thresholds

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
        B = teacher_out[0].shape[0]

        # ① L_distill
        per_layer_losses_mean = []
        per_layer_losses_inst = []
        for l in range(self.num_layers):
            s_out = student_logits_per_layer[l] / self.student_temp
            s_out = s_out.chunk(self.ncrops)
            layer_loss_mean = 0.0
            layer_loss_inst = torch.zeros(B, device=teacher_logits.device)
            n_terms = 0
            # inst_loss 추적용 카운트
            inst_terms = 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(s_out)):
                    if v == iq:
                        continue
                    
                    # chunk별 loss 계산
                    loss_v = torch.sum(-q * F.log_softmax(s_out[v], dim=-1), dim=-1)
                    
                    if v < 2:
                        # global crop일 때만 teacher 샘플 인덱스와 1:1 매칭됨
                        layer_loss_inst += loss_v[:B]
                        inst_terms += 1
                        
                    layer_loss_mean += loss_v.mean()
                    n_terms += 1
            per_layer_losses_mean.append(layer_loss_mean / n_terms)
            per_layer_losses_inst.append(layer_loss_inst / inst_terms)

        layer_weights = self._get_distill_weights(teacher_logits.device)
        L_distill = torch.zeros((), device=teacher_logits.device, dtype=teacher_logits.dtype)
        for w, loss in zip(layer_weights, per_layer_losses_mean):
            L_distill = L_distill + w * loss

        inst_losses = torch.stack(per_layer_losses_inst) # [num_layers, B]

        # ── 동적 target 생성 (로깅용 유지) ──
        layer_targets, dynamic_thresholds = self._update_ema_and_get_targets(inst_losses, epoch)

        # ③ L_align (BCE) 타겟 생성 및 계산
        B_global = teacher_logits.shape[0]
        align_terms, pred_logits_det, target_det = [], [], []

        # 하이퍼파라미터: 최소 요구 개선율 (예: 0.02 = 2% 이상 개선되지 않으면 종료)
        margin_gamma = 0.02 

        for l in range(self.num_layers):
            z_l = loss_intermediates[l][:B_global]
            pred_logit_l = loss_predictor(z_l, l).view(-1)
            
            if l == 0:
                # 첫 번째 루프는 비교 대상이 없으며, 무조건 통과해야 함 (Exit 불허)
                target_value = torch.zeros(B, device=z_l.device)
            else:
                # 한계 효용 측정: 현재 Loss가 이전 Loss의 (1 - gamma)보다 크거나 같으면 개선이 미미한 것으로 판단
                prev_loss = inst_losses[l-1].detach()
                curr_loss = inst_losses[l].detach()
                target_value = (curr_loss >= prev_loss * (1.0 - margin_gamma)).float()
            
            # 두 개의 Global Crop에 대해 타겟 복제
            target_l = target_value.repeat(2)
            
            bce_l = F.binary_cross_entropy_with_logits(
                pred_logit_l, target_l,
                pos_weight=self.bce_pos_weight_tensor.to(pred_logit_l.device)
            )
            align_terms.append(bce_l)
            pred_logits_det.append(pred_logit_l.detach())
            target_det.append(target_l.detach())

        L_align = torch.stack(align_terms).mean()

        # ③ L_pressure — expected depth 최소화
        prob_layers = []
        for l in range(self.num_layers):
            z_l = loss_intermediates[l][:B_global]
            pred_logit_l = loss_predictor(z_l, l).view(-1)
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
        alpha_pressure_curr = self.get_alpha_pressure(epoch)
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
            "alpha_pressure_curr": float(alpha_pressure_curr),
            "align_acc": align_acc.item(),
            "target_pos_rate": layer_targets.mean().item(),
            "dynamic_threshold_avg": sum(dynamic_thresholds) / len(dynamic_thresholds),
        }
        layer_pos_rates = layer_targets.mean(dim=1)  # [num_layers]
        for l in range(self.num_layers):
            loss_dict[f"pos_rate_l{l}"] = layer_pos_rates[l].item()
        return total_loss, loss_dict

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if utils.get_world_size() > 1:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * utils.get_world_size())
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
        self.heads = heads                # single shared DINOHead
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
            for l in range(num_layers):
                all_cls[l].append(cls_outs[l])
                all_loss[l].append(loss_outs[l])
            start_idx = end_idx

        cls_intermediates  = [torch.cat(feats, dim=0) for feats in all_cls]
        loss_intermediates = [torch.cat(feats, dim=0) for feats in all_loss]

        logits_per_layer = [self.heads(cls_intermediates[l]) for l in range(num_layers)]

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
    parser.add_argument('--num_indep_layers', default=4, type=int,
        help='Number of independent (unique-weight) transformer blocks. Fixed to 4 in the 4+2 design.')
    parser.add_argument('--num_shared_recurrences', default=4, type=int,
        help='Number of recurrences of the shared 2-block phase.')
    parser.add_argument('--out_dim', default=4096, type=int)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--img_size', default=224, type=int)

    # ── Loss ──
    parser.add_argument('--alpha_align', default=0.1, type=float)
    parser.add_argument('--alpha_pressure', default=0.1, type=float)
    parser.add_argument('--align_warmup_init', default=0.0, type=float)
    parser.add_argument('--align_warmup_start_epoch', default=None, type=int)
    parser.add_argument('--align_warmup_end_epoch', default=None, type=int)
    parser.add_argument('--align_warmup_mode', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--pressure_warmup_init', default=0.0, type=float)
    parser.add_argument('--pressure_warmup_start_epoch', default=None, type=int)
    parser.add_argument('--pressure_warmup_end_epoch', default=None, type=int)
    parser.add_argument('--pressure_warmup_mode', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--ema_momentum', default=0.99, type=float)
    parser.add_argument('--tau_percentile', default=0.5, type=float)
    parser.add_argument('--bce_pos_weight', default=1.0, type=float)
    parser.add_argument('--distill_mode', default='last_only', type=str,
        choices=['none', 'last_only', 'linear'],
        help='How to weight L_distill across shared recurrences.')
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
    parser.add_argument('--saveckp_freq', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    return parser


# ═════════════════════════════════════════════════════════════════════════════
# 6. Training
# ═════════════════════════════════════════════════════════════════════════════
def train_adaptive(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(vars(args).items())))

    cudnn.benchmark = True
    device = torch.device("cuda", args.gpu)

    # ── data ──
    from main_dino import DataAugmentationDINO
    transform = DataAugmentationDINO(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, shuffle=False, batch_size=args.batch_size_per_gpu,
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
        heads = DINOHead(
            embed_dim, args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        predictor = LossPredictor(embed_dim, num_shared)
        return AdaptiveMultiCropWrapper(backbone, heads, predictor)

    student = _build_model().to(device)
    teacher = _build_model().to(device)
    has_bn = utils.has_batchnorms(student)
    if has_bn:
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    student = DDP(student, device_ids=[args.gpu])
    student_without_ddp = student.module

    if has_bn:
        teacher = DDP(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    teacher_without_ddp.load_state_dict(student_without_ddp.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Hybrid model built: indep=4, shared={num_shared}, "
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
        pressure_warmup_init=args.pressure_warmup_init,
        pressure_warmup_start_epoch=args.pressure_warmup_start_epoch,
        pressure_warmup_end_epoch=args.pressure_warmup_end_epoch,
        pressure_warmup_mode=args.pressure_warmup_mode,
        ema_momentum=args.ema_momentum, tau_percentile=args.tau_percentile,
        bce_pos_weight=args.bce_pos_weight,
        distill_mode=args.distill_mode,
    ).to(device)

    # ── optimizer ──
    backbone_params = list(student_without_ddp.backbone.parameters())
    head_params = list(student_without_ddp.heads.parameters())
    predictor_params = list(student_without_ddp.loss_predictor.parameters())
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
        args.lr * args.batch_size_per_gpu * utils.get_world_size() / 256., args.min_lr,
        args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader))
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # ── resume ──
    to_restore = {"epoch": 0}
    ckp_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

        def _normalize_state_dict(state_dict, module):
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                return state_dict
            has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
            module_is_wrapped = isinstance(module, DDP)
            if has_module_prefix and not module_is_wrapped:
                return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            if module_is_wrapped and not has_module_prefix:
                return {f"module.{k}": v for k, v in state_dict.items()}
            return state_dict

        if "student" in checkpoint:
            student.load_state_dict(_normalize_state_dict(checkpoint["student"], student), strict=False)
        if "teacher" in checkpoint:
            teacher.load_state_dict(_normalize_state_dict(checkpoint["teacher"], teacher), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if fp16_scaler is not None and "fp16_scaler" in checkpoint:
            fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
        if "adaptive_loss" in checkpoint:
            adaptive_loss.load_state_dict(checkpoint["adaptive_loss"], strict=False)
        if "epoch" in checkpoint:
            to_restore["epoch"] = checkpoint["epoch"]
    start_epoch = to_restore["epoch"]

    # ── train loop ──
    start_time = time.time()
    print("Starting Hybrid SPAE training!")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student, teacher, student_without_ddp, teacher_without_ddp, adaptive_loss, data_loader,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, device)

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

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


def train_one_epoch(student, teacher, student_without_ddp, teacher_without_ddp, adaptive_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args, device):
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
                student_without_ddp.loss_predictor, loss_intermediates, epoch)

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
            for param_q, param_k in zip(student_without_ddp.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(total_loss=loss_dict["total_loss"])
        metric_logger.update(L_distill=loss_dict["L_distill"])
        metric_logger.update(L_align=loss_dict["L_align"])
        metric_logger.update(L_pressure=loss_dict["L_pressure"])
        if "alpha_align_curr" in loss_dict:
            metric_logger.update(alpha_align=loss_dict["alpha_align_curr"])
        if "alpha_pressure_curr" in loss_dict:
            metric_logger.update(alpha_pressure=loss_dict["alpha_pressure_curr"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if global_it % 50 == 0 and utils.is_main_process():
            if "align_acc" in loss_dict:
                print(f"  [BCE Predictor] acc={loss_dict['align_acc']:.4f}, "
                      f"target_pos_rate={loss_dict.get('target_pos_rate', -1):.4f}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def inference_with_early_exit(model, images, threshold=0.5):
    """threshold: sigmoid(logit) >= 값이면 exit."""
    model.eval()
    with torch.no_grad():
        wrapped = model.module if hasattr(model, "module") else model
        logits, exit_layer = wrapped.backbone.forward_inference(
            images, wrapped.loss_predictor, wrapped.heads, exit_prob=threshold)
    return logits, exit_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hybrid SPAE', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_adaptive(args)