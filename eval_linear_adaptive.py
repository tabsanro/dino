# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Linear evaluation for the adaptive DINO backbone on CIFAR-100.

This script is the adaptive-model counterpart of eval_linear.py.
It loads a checkpoint produced by main_adaptive.py, freezes the
HybridAdaptiveVisionTransformer backbone, and supports either
(1) a single linear classifier on the concatenated CLS features from
the last shared recurrences or (2) a trajectory probe with one linear
classifier per shared recurrence.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
from main_adaptive import HybridAdaptiveVisionTransformer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LinearClassifier(nn.Module):
    """Linear layer trained on top of frozen features."""

    def __init__(self, dim, num_labels=100):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class LinearProbeCollection(nn.Module):
    """Independent linear classifiers, one per shared recurrence."""

    def __init__(self, dim, num_labels=100, num_probes=6):
        super().__init__()
        self.num_labels = num_labels
        self.num_probes = num_probes
        self.probes = nn.ModuleList([LinearClassifier(dim, num_labels) for _ in range(num_probes)])

    def forward(self, features_per_probe):
        if len(features_per_probe) != len(self.probes):
            raise ValueError(
                f"Expected {len(self.probes)} feature tensors, got {len(features_per_probe)}."
            )
        return [probe(feat) for probe, feat in zip(self.probes, features_per_probe)]


def _resolve_split_path(data_path, split):
    candidate = os.path.join(data_path, split)
    return candidate if os.path.isdir(candidate) else data_path


def _build_transforms():
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


def _load_adaptive_checkpoint(pretrained_weights, checkpoint_key):
    if not pretrained_weights:
        return None
    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_weights}")
    checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    if checkpoint_key in checkpoint and isinstance(checkpoint[checkpoint_key], dict):
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        return checkpoint[checkpoint_key], checkpoint
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        print("Take key state_dict in provided checkpoint dict")
        return checkpoint["state_dict"], checkpoint
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, checkpoint
    raise ValueError(
        f"Unsupported checkpoint format in {pretrained_weights}. "
        f"Expected keys like '{checkpoint_key}', 'state_dict', or a raw state dict."
    )


def _load_backbone_weights(backbone, pretrained_weights, checkpoint_key):
    state_dict, checkpoint = _load_adaptive_checkpoint(pretrained_weights, checkpoint_key)
    if state_dict is None:
        return None

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    cleaned = {k.replace("backbone.", ""): v for k, v in cleaned.items()}
    if backbone is not None:
        model_keys = set(backbone.state_dict().keys())
        cleaned = {k: v for k, v in cleaned.items() if k in model_keys}
        msg = backbone.load_state_dict(cleaned, strict=False)
        print(f"Loaded adaptive backbone weights from {pretrained_weights} with msg: {msg}")
    return checkpoint


def _get_model_kwargs(args, checkpoint):
    ckpt_args = None
    if checkpoint is not None and isinstance(checkpoint.get("args", None), argparse.Namespace):
        ckpt_args = checkpoint["args"]

    def pick(name, default):
        value = getattr(args, name)
        if value is not None:
            return value
        if ckpt_args is not None and hasattr(ckpt_args, name):
            return getattr(ckpt_args, name)
        return default

    return {
        "img_size": pick("img_size", 224),
        "patch_size": pick("patch_size", 16),
        "embed_dim": pick("embed_dim", 384),
        "num_heads": pick("num_heads", 6),
        "drop_path_rate": pick("drop_path_rate", 0.1),
        "num_indep_layers": pick("num_indep_layers", 6),
        "num_shared_recurrences": pick("num_shared_recurrences", 6),
    }


@torch.no_grad()
def _extract_features(backbone, inp, n_last_blocks, avgpool_patchtokens=False):
    if avgpool_patchtokens:
        print("Warning: avgpool_patchtokens is ignored for the adaptive backbone; using CLS tokens only.")
    cls_intermediates, _ = backbone(inp)
    n_last_blocks = min(n_last_blocks, len(cls_intermediates))
    feats = cls_intermediates[-n_last_blocks:]
    return torch.cat(feats, dim=-1)


def _classifier_module(linear_classifier):
    return linear_classifier.module if hasattr(linear_classifier, "module") else linear_classifier


def _classifier_num_labels(linear_classifier):
    return _classifier_module(linear_classifier).num_labels


def _collect_trajectory_features(backbone, inp):
    cls_intermediates, _ = backbone(inp)
    return cls_intermediates


def _trajectory_meters(num_layers, num_labels):
    meter_map = {}
    for l in range(num_layers):
        meter_map[f"loss_l{l}"] = utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        meter_map[f"acc1_l{l}"] = utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
        if num_labels >= 5:
            meter_map[f"acc5_l{l}"] = utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    return meter_map


def _format_trajectory_summary(stats, num_layers):
    accs = [stats[f"acc1_l{l}"] for l in range(num_layers)]
    monotonic = all(accs[i] <= accs[i + 1] + 1e-12 for i in range(len(accs) - 1))
    summary = " | ".join(f"l{l} {accs[l]:.2f}" for l in range(num_layers))
    return summary + f" | monotonic={str(monotonic).lower()}"


@torch.no_grad()
def validate_network(val_loader, backbone, linear_classifier, n_last_blocks, avgpool_patchtokens, device):
    backbone.eval()
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.to(device, non_blocking=(device.type == "cuda"))
        target = target.to(device, non_blocking=(device.type == "cuda"))

        output = _extract_features(backbone, inp, n_last_blocks, avgpool_patchtokens)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if _classifier_module(linear_classifier).num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        if _classifier_module(linear_classifier).num_labels >= 5:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if _classifier_module(linear_classifier).num_labels >= 5:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
            )
        )
    else:
        print("* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, losses=metric_logger.loss
        ))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_trajectory_network(val_loader, backbone, probe_collection, device):
    backbone.eval()
    probe_collection.eval()
    num_layers = _classifier_module(probe_collection).num_probes
    num_labels = _classifier_num_labels(probe_collection)

    metric_logger = utils.MetricLogger(delimiter="  ")
    for meter_name, meter in _trajectory_meters(num_layers, num_labels).items():
        metric_logger.add_meter(meter_name, meter)
    header = "Test trajectory:"

    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.to(device, non_blocking=(device.type == "cuda"))
        target = target.to(device, non_blocking=(device.type == "cuda"))

        features_per_layer = _collect_trajectory_features(backbone, inp)
        outputs_per_layer = probe_collection(features_per_layer)

        batch_size = inp.shape[0]
        for l, output in enumerate(outputs_per_layer):
            loss = nn.CrossEntropyLoss()(output, target)
            acc1 = utils.accuracy(output, target, topk=(1,))[0]
            metric_logger.update(**{f"loss_l{l}": loss.item()})
            metric_logger.meters[f"acc1_l{l}"].update(acc1.item(), n=batch_size)
            if num_labels >= 5:
                acc5 = utils.accuracy(output, target, topk=(5,))[0]
                metric_logger.meters[f"acc5_l{l}"].update(acc5.item(), n=batch_size)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("* " + _format_trajectory_summary(stats, num_layers))
    if num_labels >= 5:
        top5_summary = " | ".join(f"l{l} {stats[f'acc5_l{l}']:.2f}" for l in range(num_layers))
        print("* Acc@5 trajectory: " + top5_summary)
    return stats


def train_one_epoch(backbone, linear_classifier, optimizer, loader, epoch, n_last_blocks, avgpool_patchtokens, device):
    backbone.eval()
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for inp, target in metric_logger.log_every(loader, 20, header):
        inp = inp.to(device, non_blocking=(device.type == "cuda"))
        target = target.to(device, non_blocking=(device.type == "cuda"))

        with torch.no_grad():
            output = _extract_features(backbone, inp, n_last_blocks, avgpool_patchtokens)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_trajectory(backbone, probe_collection, optimizer, loader, epoch, device):
    backbone.eval()
    probe_collection.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for inp, target in metric_logger.log_every(loader, 20, header):
        inp = inp.to(device, non_blocking=(device.type == "cuda"))
        target = target.to(device, non_blocking=(device.type == "cuda"))

        with torch.no_grad():
            features_per_layer = _collect_trajectory_features(backbone, inp)

        outputs_per_layer = probe_collection(features_per_layer)
        losses = [nn.CrossEntropyLoss()(output, target) for output in outputs_per_layer]
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_backbone_from_args(args, checkpoint=None):
    model_kwargs = _get_model_kwargs(args, checkpoint)
    backbone = HybridAdaptiveVisionTransformer(**model_kwargs)
    backbone.num_labels = args.num_labels
    return backbone, model_kwargs


def eval_linear_adaptive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    train_transform, val_transform = _build_transforms()
    train_root = _resolve_split_path(args.data_path, "train")
    val_root = _resolve_split_path(args.data_path, "val")

    dataset_train = datasets.ImageFolder(train_root, transform=train_transform)
    dataset_val = datasets.ImageFolder(val_root, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    checkpoint = None
    if args.pretrained_weights:
        print(f"=> loading checkpoint '{args.pretrained_weights}'")
        _, checkpoint = _load_adaptive_checkpoint(args.pretrained_weights, args.checkpoint_key)

    backbone, model_kwargs = build_backbone_from_args(args, checkpoint)
    print(
        "Adaptive backbone built with "
        f"img_size={model_kwargs['img_size']}, patch_size={model_kwargs['patch_size']}, "
        f"embed_dim={model_kwargs['embed_dim']}, indep_layers={model_kwargs['num_indep_layers']}, "
        f"shared_recurrences={model_kwargs['num_shared_recurrences']}"
    )

    if checkpoint is not None:
        _load_backbone_weights(backbone, args.pretrained_weights, args.checkpoint_key)
    else:
        print("Warning: no pretrained checkpoint provided; the backbone remains randomly initialized.")

    backbone = backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    n_last_blocks = min(args.n_last_blocks, backbone.num_shared_recurrences)
    if n_last_blocks != args.n_last_blocks:
        print(f"Clamping n_last_blocks from {args.n_last_blocks} to {n_last_blocks}.")

    trajectory_mode = args. probe_trajectory
    if trajectory_mode:
        probe_layers = backbone.num_shared_recurrences
        linear_classifier = LinearProbeCollection(backbone.embed_dim, num_labels=args.num_labels, num_probes=probe_layers).to(device)
        print(f"Trajectory probing enabled with {probe_layers} independent linear heads.")
    else:
        embed_dim = backbone.embed_dim * n_last_blocks
        linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels).to(device)

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * args.batch_size_per_gpu / 256.0,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    to_restore = {"epoch": 0, "best_acc": ([0.0] * backbone.num_shared_recurrences if trajectory_mode else 0.0)}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    if args.evaluate:
        if trajectory_mode:
            test_stats = validate_trajectory_network(val_loader, backbone, linear_classifier, device)
            print(_format_trajectory_summary(test_stats, _classifier_module(linear_classifier).num_probes))
        else:
            test_stats = validate_network(val_loader, backbone, linear_classifier, n_last_blocks, args.avgpool_patchtokens, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    for epoch in range(start_epoch, args.epochs):
        if trajectory_mode:
            train_stats = train_one_epoch_trajectory(
                backbone,
                linear_classifier,
                optimizer,
                train_loader,
                epoch,
                device,
            )
        else:
            train_stats = train_one_epoch(
                backbone,
                linear_classifier,
                optimizer,
                train_loader,
                epoch,
                n_last_blocks,
                args.avgpool_patchtokens,
                device,
            )
        scheduler.step()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            if trajectory_mode:
                test_stats = validate_trajectory_network(val_loader, backbone, linear_classifier, device)
                layer_accs = [test_stats[f"acc1_l{l}"] for l in range(_classifier_module(linear_classifier).num_probes)]
                if isinstance(best_acc, list):
                    best_acc = [max(best_acc[l], layer_accs[l]) for l in range(len(layer_accs))]
                else:
                    best_acc = layer_accs
                print(f"Accuracy trajectory at epoch {epoch}: {', '.join(f'l{l} {layer_accs[l]:.1f}%' for l in range(len(layer_accs)))}")
                print(f"Best trajectory so far: {', '.join(f'l{l} {best_acc[l]:.1f}%' for l in range(len(best_acc)))}")
            else:
                test_stats = validate_network(val_loader, backbone, linear_classifier, n_last_blocks, args.avgpool_patchtokens, device)
                print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                best_acc = max(best_acc, test_stats["acc1"])
                print(f"Max accuracy so far: {best_acc:.2f}%")
            log_stats = {**log_stats, **{f"test_{k}": v for k, v in test_stats.items()}}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

    print(
        "Training of the supervised linear classifier on frozen adaptive features completed.\n"
        + (
            "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc)
            if not trajectory_mode
            else "Trajectory best accuracies: " + ", ".join(f"l{l} {best_acc[l]:.1f}" for l in range(len(best_acc)))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Adaptive linear evaluation on CIFAR-100")

    # Backbone configuration. Leave as None to auto-read values from the checkpoint if available.
    parser.add_argument("--img_size", default=None, type=int)
    parser.add_argument("--patch_size", default=None, type=int)
    parser.add_argument("--embed_dim", default=None, type=int)
    parser.add_argument("--num_heads", default=None, type=int)
    parser.add_argument("--drop_path_rate", default=None, type=float)
    parser.add_argument("--num_indep_layers", default=None, type=int)
    parser.add_argument("--num_shared_recurrences", default=None, type=int)

    # Linear-eval settings
    parser.add_argument("--n_last_blocks", default=4, type=int, help="Concatenate the CLS tokens from the last n shared recurrences.")
    parser.add_argument("--avgpool_patchtokens", default=False, type=utils.bool_flag, help="Kept for CLI compatibility; ignored by the adaptive backbone.")
    parser.add_argument("--pretrained_weights", default="", type=str, help="Path to a checkpoint produced by main_adaptive.py.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help="Key to use inside the checkpoint dict (teacher or student).")
    parser.add_argument("--probe_trajectory", action="store_true", help="Train one linear classifier per shared recurrence and report the per-layer accuracy trajectory.")

    # Training
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size_per_gpu", default=128, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--val_freq", default=1, type=int)
    parser.add_argument("--output_dir", default="./output_linear_adaptive", type=str)
    parser.add_argument("--num_labels", default=100, type=int)
    parser.add_argument("--evaluate", dest="evaluate", action="store_true")

    # Data
    parser.add_argument("--data_path", default="./cifar100_images", type=str, help="Root folder containing train/ and val/ ImageFolder splits.")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear_adaptive(args)
