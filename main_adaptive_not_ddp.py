"""
Self-Predictive Adaptive Encoder (SPAE) - Non-DDP Training Script
=================================================================
main_adaptive.py를 기반으로, DDP 없이 단일 프로세스(1 GPU 또는 CPU)에서
3가지 Loss (Distill + Align + Pressure)를 사용하여
Adaptive Computational Encoder를 학습합니다.

Usage:
    python main_adaptive_not_ddp.py \
        --data_path /path/to/cifar100 --epochs 100 --num_recurrences 10
"""
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torchvision import datasets

import utils
from vision_transformer import DINOHead
from adaptive_encoder import (
    AdaptiveVisionTransformer,
    LossPredictor,
    AdaptiveDINOLoss,
    AdaptiveMultiCropWrapper,
)


def get_args_parser():
    parser = argparse.ArgumentParser('SPAE - Self-Predictive Adaptive Encoder (Non-DDP)', add_help=False)

    # ── Model parameters ──
    parser.add_argument('--embed_dim', default=384, type=int,
        help='Embedding dimension for the shared block.')
    parser.add_argument('--num_heads', default=6, type=int,
        help='Number of attention heads in the shared block.')
    parser.add_argument('--patch_size', default=16, type=int,
        help='Patch size for patch embedding.')
    parser.add_argument('--num_recurrences', default=10, type=int,
        help='Number of times the shared block is applied (depth).')
    parser.add_argument('--out_dim', default=4096, type=int,
        help='Output dimension of DINO heads. 4096 for CIFAR, 65536 for ImageNet.')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--img_size', default=224, type=int)

    # ── Loss parameters ──
    parser.add_argument('--alpha_align', default=0.1, type=float,
        help='Target weight for L_align after warmup.')
    parser.add_argument('--alpha_pressure', default=0.1, type=float,
        help='Weight for L_pressure (efficiency pressure). KEY HYPERPARAMETER.')
    parser.add_argument('--align_warmup_init', default=0.0, type=float,
        help='Initial alpha_align before warmup starts.')
    parser.add_argument('--align_warmup_start_epoch', default=30, type=int,
        help='Epoch to start increasing alpha_align.')
    parser.add_argument('--align_warmup_end_epoch', default=60, type=int,
        help='Epoch to reach target alpha_align.')
    parser.add_argument('--align_warmup_mode', default='cosine', type=str, choices=['cosine', 'linear'],
        help='Warmup schedule type for alpha_align.')

    parser.add_argument('--exit_tau', default=0.1, type=float,
        help='loss <= tau 이면 early-exit positive(1) 라벨.')
    parser.add_argument('--bce_pos_weight', default=1.0, type=float,
        help='BCE positive class weight.')
    parser.add_argument('--early_exit_prob', default=0.5, type=float,
        help='inference에서 sigmoid(logit) >= 값이면 exit.')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int)
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
        help='Base EMA parameter for teacher update.')

    # ── Training parameters ──
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # ── Multi-crop parameters ──
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))

    # ── Misc ──
    parser.add_argument('--data_path', default='./cifar100_images/train', type=str)
    parser.add_argument('--output_dir', default="./output_adaptive", type=str)
    parser.add_argument('--saveckp_freq', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    return parser


def train_adaptive(args):
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    # ============ preparing data ... ============
    from main_dino import DataAugmentationDINO
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building SPAE model ... ============
    ncrops = args.local_crops_number + 2
    embed_dim = args.embed_dim
    num_layers = args.num_recurrences

    student_backbone = AdaptiveVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=embed_dim,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path_rate,
        num_recurrences=num_layers,
    )

    student_heads = nn.ModuleList([
        DINOHead(embed_dim, args.out_dim,
                 use_bn=args.use_bn_in_head,
                 norm_last_layer=args.norm_last_layer if l == num_layers - 1 else True)
        for l in range(num_layers)
    ])

    student_loss_predictor = LossPredictor(embed_dim, num_layers)
    student = AdaptiveMultiCropWrapper(student_backbone, student_heads, student_loss_predictor).to(device)
    
    teacher_backbone = AdaptiveVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=embed_dim,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path_rate,
        num_recurrences=num_layers,
    )

    teacher_heads = nn.ModuleList([
        DINOHead(embed_dim, args.out_dim,
                 use_bn=args.use_bn_in_head,
                 norm_last_layer=args.norm_last_layer if l == num_layers - 1 else True)
        for l in range(num_layers)
    ])

    teacher_loss_predictor = LossPredictor(embed_dim, num_layers)
    teacher = AdaptiveMultiCropWrapper(teacher_backbone, teacher_heads, teacher_loss_predictor).to(device)

    # teacher initialized with student weights
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"SPAE model built: embed_dim={embed_dim}, num_recurrences={num_layers}, device={device}")

    # ============ preparing loss ... ============
    adaptive_loss = AdaptiveDINOLoss(
        out_dim=args.out_dim,
        ncrops=ncrops,
        num_layers=num_layers,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        alpha_align=args.alpha_align,
        alpha_pressure=args.alpha_pressure,
        align_warmup_init=args.align_warmup_init,
        align_warmup_start_epoch=args.align_warmup_start_epoch,
        align_warmup_end_epoch=args.align_warmup_end_epoch,
        align_warmup_mode=args.align_warmup_mode,
        exit_tau=args.exit_tau,
        bce_pos_weight=args.bce_pos_weight,
    ).to(device)

    # ============ preparing optimizer ... ============
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

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size_per_gpu / 256.,
        args.min_lr, args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher, optimizer=optimizer,
        fp16_scaler=fp16_scaler, adaptive_loss=adaptive_loss,
    )
    start_epoch = to_restore["epoch"]

    # ============ training loop ... ============
    start_time = time.time()
    print("Starting SPAE training (Non-DDP)!")
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch_adaptive(
            student, teacher, adaptive_loss, data_loader,
            optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args, device)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'adaptive_loss': adaptive_loss.state_dict(),
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


def train_one_epoch_adaptive(student, teacher, adaptive_loss,
                             data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                             epoch, fp16_scaler, args, device):
    """
    학습 시에는 Early Exit 없이 모든 블록을 통과하며 3가지 Loss를 최적화.
    """
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
            # teacher forward passes (only 2 global views)
            teacher_logits_per_layer, _ = teacher(images[:2])
            teacher_logits = teacher_logits_per_layer[-1]
            
            # student forward passes
            logits_per_layer, intermediates = student(images)

            loss, loss_dict = adaptive_loss(
                logits_per_layer, teacher_logits,
                student.loss_predictor,
                intermediates, epoch)

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

        # EMA update for the teacher
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

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def inference_with_early_exit(model_without_ddp, images, threshold=0.5):
    """
    threshold: sigmoid(logit) 확률 임계값
    """
    model_without_ddp.eval()
    with torch.no_grad():
        logits, exit_layer = model_without_ddp.backbone.forward_inference(
            images,
            model_without_ddp.loss_predictor,
            model_without_ddp.heads,
            exit_prob=threshold,
        )
    return logits, exit_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPAE (Non-DDP)', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_adaptive(args)