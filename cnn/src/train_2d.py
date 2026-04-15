"""
train_2d.py
==================

Training pipeline for fault detection in 2D marine seismic sections.
Uses Faster R-CNN with FPN backbone (ResNet-50) and PyTorch.

We went with Faster R-CNN over YOLO/SSD because faults are tiny compared
to the full seismic image, and two-stage detectors handle small objects
better. The FPN gives us feature maps at multiple scales so we can catch
faults whether they're 20px or 200px. ResNet-50 is well matched to this
dataset size (~168 images) — ResNet-101 has more capacity but risks
overfitting with limited data and is significantly slower on MPS.

MPS memory is capped at 70% of unified memory (≈17 GB on a 24 GB M4 Pro)
via PYTORCH_MPS_HIGH_WATERMARK_RATIO so the OS and other apps stay
responsive during overnight runs.

EXPECTED DATA STRUCTURE:
    data/
    ├── images/          ← can be overridden with --images_dir
    │   ├── T2010.100.mig.1500.png
    │   └── ...
    ├── train.json
    └── val.json

USAGE:
    python train_2d.py \\
        --data_dir ../../pipeline/merged \\
        --images_dir ../../pipeline/results/run_20260403_093551/images

    # Resume from checkpoint:
    python train_2d.py --data_dir ./data --resume checkpoints/best_model.pth

    # Inference on new image:
    python train_2d.py --predict --image path/to/seismic.png --weights checkpoints/best_model.pth
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

# Cap MPS at 70% of unified memory — leaves ~7 GB for macOS and other apps.
# Must be set before importing torch.

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    # --- Model ---
    "backbone": "resnet50",  # resnet50 or resnet101
    "pretrained_backbone": True,  # ImageNet pretrained (critical for small data)
    "num_classes": 2,  # 1 class (fault) + background
    "min_size": 1024,  # Min image dimension during training
    "max_size": 2048,  # Max image dimension during training
    # --- RPN Anchor sizes ---
    # Tuned for fault shapes: tall/narrow boxes at multiple scales
    "anchor_sizes": ((16,), (32,), (64,), (128,), (256,)),
    "anchor_aspect_ratios": ((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    # --- Training ---
    "learning_rate": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler_step": 30,  # Scaled from 40 to suit 80-epoch run
    "lr_scheduler_gamma": 0.5,
    "warmup_epochs": 3,
    "epochs": 80,  # ~4-5 hrs on M4 Pro MPS with resnet50 + batch 1
    "batch_size": 1,  # Keeps memory pressure low, leaves OS headroom
    "num_workers": 0,  # Required for MPS on macOS
    # --- Augmentation ---
    "augment_horizontal_flip": True,
    "augment_brightness": 0.2,
    "augment_contrast": 0.2,
    "augment_crop_prob": 0.5,
    "augment_crop_min_frac": 0.6,
    # --- Evaluation ---
    "iou_threshold": 0.3,
    "score_threshold": 0.95,
    "nms_threshold": 0.3,
    # --- Checkpointing ---
    "save_every": 10,
    "checkpoint_dir": "checkpoints",
}

# Print a live status line every N batches so you can confirm training is running
HEARTBEAT_EVERY = 5


# =============================================================================
# HEARTBEAT
# =============================================================================


def heartbeat(batch_idx, n_batches, epoch, loss, epoch_start):
    """Print a timestamped status line every HEARTBEAT_EVERY batches."""
    batches_done = batch_idx + 1
    batches_left = n_batches - batches_done
    elapsed = time.time() - epoch_start
    rate = elapsed / batches_done if batches_done > 0 else 0
    eta = timedelta(seconds=int(rate * batches_left))
    now = datetime.now().strftime("%H:%M:%S")
    print(
        f"  [{now}]  epoch {epoch}  batch {batches_done}/{n_batches}  "
        f"loss {loss:.4f}  ETA this epoch: {eta}",
        flush=True,
    )


# =============================================================================
# DATASET
# =============================================================================


class SeismicFaultDataset(torch.utils.data.Dataset):
    """
    Dataset for seismic fault detection.
    Loads PNG images and COCO-format JSON annotations.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the annotation JSON files.
    annotation_file : str
        Filename of the COCO JSON (e.g. "train.json").
    transforms : callable, optional
        Transform to apply to (image, target) pairs.
    image_dir : str or Path, optional
        Directory containing the PNG images. Defaults to data_dir/images/.
    """

    def __init__(self, data_dir, annotation_file, transforms=None, image_dir=None):
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / "images"
        self.transforms = transforms

        ann_path = self.data_dir / annotation_file
        with open(ann_path, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.image_ids = list(self.images.keys())

        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        print(
            f"  Loaded {len(self.image_ids)} images, "
            f"{len(self.coco['annotations'])} annotations from {annotation_file}"
        )
        print(f"  Image dir: {self.image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = self.image_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])
        boxes, labels, areas = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(w * h)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# =============================================================================
# AUGMENTATION
# =============================================================================


class SeismicAugmentation:
    def __init__(self, config, train=True):
        self.config = config
        self.train = train

    def __call__(self, image, target):
        image = T.functional.to_tensor(image)

        if self.train:
            if self.config["augment_horizontal_flip"] and random.random() < 0.5:
                image = T.functional.hflip(image)
                if len(target["boxes"]) > 0:
                    w = image.shape[2]
                    boxes = target["boxes"]
                    boxes_flipped = boxes.clone()
                    boxes_flipped[:, 0] = w - boxes[:, 2]
                    boxes_flipped[:, 2] = w - boxes[:, 0]
                    target["boxes"] = boxes_flipped

            if self.config["augment_brightness"] > 0:
                factor = 1.0 + random.uniform(
                    -self.config["augment_brightness"],
                    self.config["augment_brightness"],
                )
                image = T.functional.adjust_brightness(image, factor)

            if self.config["augment_contrast"] > 0:
                factor = 1.0 + random.uniform(
                    -self.config["augment_contrast"],
                    self.config["augment_contrast"],
                )
                image = T.functional.adjust_contrast(image, factor)

            if random.random() < self.config["augment_crop_prob"]:
                image, target = self._random_crop(image, target)

        return image, target

    def _random_crop(self, image, target):
        _, h, w = image.shape
        min_frac = self.config["augment_crop_min_frac"]
        crop_h = random.randint(int(h * min_frac), h)
        crop_w = random.randint(int(w * min_frac), w)

        if len(target["boxes"]) > 0:
            box_idx = random.randint(0, len(target["boxes"]) - 1)
            box = target["boxes"][box_idx]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            x1 = int(
                max(0, cx - crop_w / 2 + random.randint(-crop_w // 4, crop_w // 4))
            )
            y1 = int(
                max(0, cy - crop_h / 2 + random.randint(-crop_h // 4, crop_h // 4))
            )
        else:
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

        x1 = min(x1, w - crop_w)
        y1 = min(y1, h - crop_h)
        x2, y2 = x1 + crop_w, y1 + crop_h
        image = image[:, y1:y2, x1:x2]

        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] -= x1
            boxes[:, 1] -= y1
            boxes[:, 2] -= x1
            boxes[:, 3] -= y1
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=crop_w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=crop_h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=crop_w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=crop_h)

            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths > 5) & (heights > 5)

            target["boxes"] = boxes[valid]
            target["labels"] = target["labels"][valid]
            target["area"] = (boxes[valid, 2] - boxes[valid, 0]) * (
                boxes[valid, 3] - boxes[valid, 1]
            )
            target["iscrowd"] = target["iscrowd"][valid]

        return image, target


# =============================================================================
# MODEL
# =============================================================================


def build_model(config):
    print("\n  Building Faster R-CNN + FPN...")

    if config["backbone"] == "resnet101":
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name="resnet101",
            weights="DEFAULT" if config["pretrained_backbone"] else None,
        )
        print("    Backbone: ResNet-101 + FPN (pretrained)")
    else:
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name="resnet50",
            weights="DEFAULT" if config["pretrained_backbone"] else None,
        )
        print("    Backbone: ResNet-50 + FPN (pretrained)")

    anchor_generator = AnchorGenerator(
        sizes=config["anchor_sizes"],
        aspect_ratios=config["anchor_aspect_ratios"],
    )

    model = FasterRCNN(
        backbone,
        num_classes=config["num_classes"],
        rpn_anchor_generator=anchor_generator,
        min_size=config["min_size"],
        max_size=config["max_size"],
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        box_score_thresh=config["score_threshold"],
        box_nms_thresh=config["nms_threshold"],
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    return model


# =============================================================================
# TRAINING
# =============================================================================


def train_one_epoch(model, optimizer, data_loader, device, epoch, config):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"    WARNING: NaN loss at batch {batch_idx}, skipping", flush=True)
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1
        running_avg = total_loss / n_batches

        # Heartbeat — printed every HEARTBEAT_EVERY batches so you can see it's alive
        if (batch_idx + 1) % HEARTBEAT_EVERY == 0:
            heartbeat(batch_idx, len(data_loader), epoch + 1, running_avg, epoch_start)

        # Full loss breakdown every 20 batches and at end of epoch
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(data_loader):
            detail = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(
                f"    Batch {batch_idx+1}/{len(data_loader)} | "
                f"Loss: {running_avg:.4f} | {detail}",
                flush=True,
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, tgt in zip(predictions, targets):
            all_predictions.append(
                {
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu(),
                }
            )
            all_targets.append(
                {
                    "boxes": tgt["boxes"],
                    "labels": tgt["labels"],
                }
            )

    iou_thresh = config["iou_threshold"]
    total_tp = total_fp = total_fn = 0

    for pred, tgt in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        elif len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        elif len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        score_order = pred_scores.argsort(descending=True)

        for pred_idx in score_order:
            best_iou, best_gt = ious[pred_idx].max(dim=0)
            if best_iou >= iou_thresh and best_gt.item() not in matched_gt:
                total_tp += 1
                matched_gt.add(best_gt.item())
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def collate_fn(batch):
    return tuple(zip(*batch))


def train(config, data_dir, images_dir=None):
    print("\n" + "=" * 65)
    print("  FAULT DETECTION CNN — TRAINING PIPELINE")
    print("=" * 65)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n  Device:          {device}")
    print(
        f"  MPS memory cap:  {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default')}"
    )
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print(f"  Started at:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Rough ETA shown upfront so you know when to check back
    secs_per_epoch = 55  # conservative for resnet50 + batch 1 on M4 Pro MPS
    est_finish = datetime.now() + timedelta(seconds=secs_per_epoch * config["epochs"])
    print(
        f"  Estimated finish: ~{est_finish.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({config['epochs']} epochs × ~{secs_per_epoch}s each)"
    )

    print("\n  Loading datasets...")
    train_dataset = SeismicFaultDataset(
        data_dir,
        "train.json",
        transforms=SeismicAugmentation(config, train=True),
        image_dir=images_dir,
    )
    val_dataset = SeismicFaultDataset(
        data_dir,
        "val.json",
        transforms=SeismicAugmentation(config, train=False),
        image_dir=images_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_model(config)
    model.to(device)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": config["learning_rate"] * 0.1},
            {"params": head_params, "lr": config["learning_rate"]},
        ],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_scheduler_step"],
        gamma=config["lr_scheduler_gamma"],
    )

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    best_f1 = 0.0
    start_epoch = 0

    if config.get("resume"):
        print(f"\n  Resuming from: {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best F1 so far: {best_f1:.3f}")

    print(f"\n  Epochs:      {config['epochs']}")
    print(f"  Train images: {len(train_dataset)}  |  Val images: {len(val_dataset)}")
    print(f"  Batch size:  {config['batch_size']}  |  LR: {config['learning_rate']}")
    print(f"  Heartbeat every {HEARTBEAT_EVERY} batches")
    print("-" * 65, flush=True)

    history = {"train_loss": [], "val_precision": [], "val_recall": [], "val_f1": []}
    training_start = time.time()

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()

        if epoch < config["warmup_epochs"]:
            warmup_factor = (epoch + 1) / config["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_factor

        print(
            f"\n  Epoch {epoch+1}/{config['epochs']}  "
            f"[{datetime.now().strftime('%H:%M:%S')}]",
            flush=True,
        )

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, config
        )
        val_metrics = evaluate(model, val_loader, device, config)

        if epoch >= config["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - epoch_start
        epochs_left = config["epochs"] - (epoch + 1)
        eta_remaining = timedelta(seconds=int(elapsed * epochs_left))
        current_lr = optimizer.param_groups[1]["lr"]

        print(
            f"    Loss: {train_loss:.4f} | "
            f"P: {val_metrics['precision']:.3f} | "
            f"R: {val_metrics['recall']:.3f} | "
            f"F1: {val_metrics['f1']:.3f} | "
            f"TP:{val_metrics['tp']} FP:{val_metrics['fp']} FN:{val_metrics['fn']} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s | "
            f"ETA remaining: {eta_remaining}",
            flush=True,
        )

        history["train_loss"].append(train_loss)
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "config": config,
                },
                ckpt_dir / "best_model.pth",
            )
            print(f"    *** New best F1: {best_f1:.3f} — saved ***", flush=True)

        if (epoch + 1) % config["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "config": config,
                },
                ckpt_dir / f"checkpoint_epoch_{epoch+1}.pth",
            )
            print(f"    Checkpoint saved: checkpoint_epoch_{epoch+1}.pth", flush=True)

    plot_training_history(history, ckpt_dir / "training_history.png")
    print(f"\n  Training complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Best F1: {best_f1:.3f}")
    return model, history


# =============================================================================
# INFERENCE
# =============================================================================


@torch.no_grad()
def predict(image_path, weights_path, config=None, output_path=None):
    if config is None:
        config = TRAIN_CONFIG

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(config)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = T.functional.to_tensor(image).to(device)
    predictions = model([image_tensor])[0]

    keep = predictions["scores"] >= config["score_threshold"]
    boxes = predictions["boxes"][keep].cpu().numpy()
    scores = predictions["scores"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()

    print(f"\n  Predictions: {len(boxes)} faults detected")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(
            f"    Fault {i+1}: box=[{box[0]:.0f}, {box[1]:.0f}, "
            f"{box[2]:.0f}, {box[3]:.0f}] confidence={score:.3f}"
        )

    if output_path:
        visualize_predictions(image, boxes, scores, output_path)

    return boxes, scores, labels


def visualize_predictions(image, boxes, scores, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(image)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = "#00FFFF" if score >= 0.8 else "#00FF66" if score >= 0.6 else "#FFFF00"

        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.15,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
        )
        ax.text(
            x1,
            y1 - 5,
            f"F{i+1} ({score:.2f})",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="black", alpha=0.8, edgecolor=color
            ),
        )

    ax.set_title("CNN Fault Predictions", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")


# =============================================================================
# UTILITIES
# =============================================================================


def plot_training_history(history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], "b-", lw=1.5)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_precision"], "g-", lw=1.5, label="Precision")
    axes[1].plot(history["val_recall"], "r-", lw=1.5, label="Recall")
    axes[1].set_title("Validation Precision & Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_f1"], "m-", lw=1.5)
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved training history: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fault Detection CNN — Train & Predict"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing train.json and val.json",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing PNG images (default: data_dir/images/)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--backbone", type=str, default=None, choices=["resnet50", "resnet101"]
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output", type=str, default="predictions.png")

    args = parser.parse_args()

    config = TRAIN_CONFIG.copy()
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.backbone:
        config["backbone"] = args.backbone
    if args.resume:
        config["resume"] = args.resume

    if args.predict:
        if not args.image or not args.weights:
            print("ERROR: --predict requires --image and --weights")
            sys.exit(1)
        predict(args.image, args.weights, config, args.output)
    else:
        train(config, args.data_dir, images_dir=args.images_dir)


if __name__ == "__main__":
    main()
