"""
Self-Predictive Adaptive Encoder (SPAE)
========================================
DINO의 Self-Distillation 구조를 기반으로 한 Adaptive Computational Encoder.

핵심 아이디어:
- 하나의 공유 Transformer Block을 N번 반복 (Recurrent)
- 각 레이어에서 Loss Predictor가 "내가 지금 얼마나 틀릴지"를 예측
- Efficiency Pressure로 앞쪽 레이어에서도 높은 품질의 표현을 학습하도록 강제
- 인퍼런스 시 predicted loss < threshold 이면 Early Exit
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vision_transformer import Block, PatchEmbed, DINOHead
from utils import trunc_normal_


# =============================================================================
# 1. Adaptive Vision Transformer (Shared Block, Recurrent)
# =============================================================================
class AdaptiveVisionTransformer(nn.Module):
    """
    파라미터를 공유하는 하나의 Block을 num_recurrences번 반복 사용.
    각 반복(레이어)의 CLS token을 중간 표현으로 반환.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, num_recurrences=10, **kwargs):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_recurrences = num_recurrences

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 공유 Transformer Block (1개)
        self.shared_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer)

        # 각 반복에서 사용할 Layer Norm (공유하지 않음 - 각 깊이의 분포가 다르므로)
        self.norms = nn.ModuleList([norm_layer(embed_dim) for _ in range(num_recurrences)])

        # ImageNet classification head 호환용
        self.head = nn.Identity()
        self.fc = nn.Identity()

        self.step_embeddings = nn.Parameter(torch.zeros(self.num_recurrences, 1, 1, embed_dim))
        trunc_normal_(self.step_embeddings, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        """
        Returns:
            intermediates: list of [B, embed_dim] CLS tokens, one per recurrence
        """
        x = self.prepare_tokens(x)
        intermediates = []
        for l in range(self.num_recurrences):
            x = self.shared_block(x)
            cls_out = self.norms[l](x)[:, 0]  # [B, embed_dim]
            intermediates.append(cls_out)
        return intermediates

    def forward_inference(self, images, loss_predictor, heads, exit_prob=0.5):
        """
        인퍼런스 시 Early Exit:
        sigmoid(logit) >= exit_prob 이면 exit.
        Returns:
            logits: [B, out_dim], exit_layer: int (0-indexed)
        """
        x = self.prepare_tokens(images)
        for l in range(self.num_recurrences):
            x = x + self.step_embeddings[l]
            x = self.shared_block(x)
            cls_out = self.norms[l](x)[:, 0]

            with torch.no_grad():
                pred_logit = loss_predictor(cls_out, l)   # [B]
                pred_prob = torch.sigmoid(pred_logit)     # [B]

            # 현재 구현은 배치 단위로 같은 레이어에서 종료
            if torch.all(pred_prob >= exit_prob):
                logits = heads[l](cls_out)
                return logits, l

        logits = heads[-1](cls_out)
        return logits, self.num_recurrences - 1


# =============================================================================
# 2. Loss Predictor R: 각 레이어의 예상 DINO Loss를 예측하는 MLP
# =============================================================================
class LossPredictor(nn.Module):
    """
    각 레이어의 CLS feature -> early-exit 가능성(logit) 예측.
    (회귀값이 아니라 BCE용 logit 출력)
    """
    def __init__(self, embed_dim, num_layers=10, hidden_dim=128):
        super().__init__()
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),  # logit (no activation)
            )
            for _ in range(num_layers)
        ])

    def forward(self, z, layer_idx):
        """
        Args:
            z: [B, embed_dim]
            layer_idx: int
        Returns:
            pred_logit: [B]
        """
        return self.predictors[layer_idx](z).squeeze(-1)


# =============================================================================
# 3. Adaptive DINO Loss: 3가지 Loss의 통합
# =============================================================================
class AdaptiveDINOLoss(nn.Module):
    """
    ① L_distill
    ② L_align (BCE)
    ③ L_pressure
    + alpha_align warmup curriculum
    """
    def __init__(self, out_dim, ncrops, num_layers=10,
                 warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0, nepochs=100,
                 student_temp=0.1, center_momentum=0.9,
                 alpha_align=0.1, alpha_pressure=0.1,
                 align_warmup_init=0.0,
                 align_warmup_start_epoch=30,
                 align_warmup_end_epoch=60,
                 align_warmup_mode="cosine",
                 exit_tau=0.1, bce_pos_weight=1.0):
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

        self.exit_tau = exit_tau
        self.register_buffer("bce_pos_weight_tensor", torch.tensor(float(bce_pos_weight), dtype=torch.float32))

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def get_alpha_align(self, epoch: int) -> float:
        s = self.align_warmup_start_epoch
        e = self.align_warmup_end_epoch
        init = self.align_warmup_init
        target = self.alpha_align_target

        if e <= s:
            return target if epoch >= s else init
        if epoch < s:
            return init
        if epoch >= e:
            return target

        t = (epoch - s) / float(e - s)  # [0,1]
        if self.align_warmup_mode == "linear":
            w = t
        else:  # cosine
            w = 0.5 * (1.0 - math.cos(math.pi * t))
        return init + (target - init) * w

    def forward(self, student_logits_per_layer, teacher_logits,
                loss_predictor, intermediates, epoch):
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        teacher_out = F.softmax((teacher_logits - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # ① L_distill
        per_layer_losses = []
        for l in range(self.num_layers):
            s_out = student_logits_per_layer[l] / self.student_temp
            s_out = s_out.chunk(self.ncrops)

            layer_loss = 0.0
            n_terms = 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(s_out)):
                    if v == iq:
                        continue
                    loss = torch.sum(-q * F.log_softmax(s_out[v], dim=-1), dim=-1)
                    layer_loss += loss.mean()
                    n_terms += 1
            per_layer_losses.append(layer_loss / n_terms)

        L_distill = sum(per_layer_losses[:self.num_layers - 1])

        # ② L_align (BCE): target = 1(loss <= tau)
        B_global = teacher_logits.shape[0]  # 2*B
        align_terms = []
        pred_logits_layers_detached = []
        target_layers_detached = []

        for l in range(self.num_layers):
            z_l_global = intermediates[l][:B_global]                       # [2B, D]
            pred_logit_l = loss_predictor(z_l_global.detach(), l).view(-1) # [2B]

            target_value = 1.0 if per_layer_losses[l].detach().item() <= self.exit_tau else 0.0
            target_l = torch.full_like(pred_logit_l, target_value)

            bce_l = F.binary_cross_entropy_with_logits(
                pred_logit_l,
                target_l,
                pos_weight=self.bce_pos_weight_tensor.to(pred_logit_l.device),
            )
            align_terms.append(bce_l)
            pred_logits_layers_detached.append(pred_logit_l.detach())
            target_layers_detached.append(target_l.detach())

        L_align = torch.stack(align_terms).mean()

        # ③ L_pressure: expected depth 최소화
        prob_layers = []
        for l in range(self.num_layers):
            z_l_global = intermediates[l][:B_global]               # [2B, D]
            pred_logit_l = loss_predictor(z_l_global, l).view(-1)  # [2B]
            prob_layers.append(torch.sigmoid(pred_logit_l))        # [2B]

        p = torch.stack(prob_layers, dim=1)  # [2B, L]
        B, L = p.shape
        surv = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=p.device), (1.0 - p + 1e-6)], dim=1),
            dim=1
        )[:, :-1]
        exit_dist = surv * p
        exit_dist[:, -1] += (1.0 - exit_dist.sum(dim=1))
        depths = torch.arange(1, L + 1, device=p.device, dtype=p.dtype).unsqueeze(0)
        expected_depth = (exit_dist * depths).sum(dim=1)
        L_pressure = (expected_depth / L).mean()

        alpha_align_curr = self.get_alpha_align(epoch)
        total_loss = L_distill + alpha_align_curr * L_align + self.alpha_pressure * L_pressure

        self.update_center(teacher_logits)

        with torch.no_grad():
            pred_all = torch.cat(pred_logits_layers_detached, dim=0)
            tgt_all = torch.cat(target_layers_detached, dim=0)
            align_acc = ((pred_all >= 0).float() == tgt_all).float().mean()
            target_pos_rate = tgt_all.mean()
            predicted_probs = [torch.sigmoid(x).mean().item() for x in pred_logits_layers_detached]

        loss_dict = {
            "L_distill": L_distill.item(),
            "L_align": L_align.item(),
            "L_pressure": L_pressure.item(),
            "total_loss": total_loss.item(),
            "alpha_align_curr": float(alpha_align_curr),
            "align_acc": align_acc.item(),
            "target_pos_rate": target_pos_rate.item(),
            "per_layer_losses": [l.item() for l in per_layer_losses],
            "predicted_probs": predicted_probs,
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# =============================================================================
# 4. Multi-Crop Wrapper (Adaptive 버전)
# =============================================================================
class AdaptiveMultiCropWrapper(nn.Module):
    """
    Multi-resolution crop을 처리하고, 각 레이어별 head 출력과
    중간 features를 반환하는 wrapper.
    """
    def __init__(self, backbone, heads, loss_predictor):
        super().__init__()
        self.backbone = backbone
        self.heads = heads  # nn.ModuleList of DINOHead
        self.loss_predictor = loss_predictor

    def forward(self, x):
        """
        Args:
            x: list of tensors (multi-crop augmented views)
        Returns:
            logits_per_layer: list[Tensor] - 각 레이어의 head 출력
            intermediates: list[Tensor] - 각 레이어의 CLS features
        """
        if not isinstance(x, list):
            x = [x]

        # 같은 해상도끼리 묶어서 forward
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        num_layers = self.backbone.num_recurrences
        all_intermediates = [[] for _ in range(num_layers)]

        start_idx = 0
        for end_idx in idx_crops:
            batch = torch.cat(x[start_idx:end_idx])
            layer_outputs = self.backbone(batch)  # list of [B_chunk, embed_dim]
            for l in range(num_layers):
                all_intermediates[l].append(layer_outputs[l])
            start_idx = end_idx

        # 각 레이어별 concat
        intermediates = [torch.cat(feats, dim=0) for feats in all_intermediates]

        # 각 레이어별 head 적용
        logits_per_layer = [self.heads[l](intermediates[l]) for l in range(num_layers)]

        return logits_per_layer, intermediates

