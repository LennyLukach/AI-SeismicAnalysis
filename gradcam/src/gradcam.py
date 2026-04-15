# gradcam_all.py
import torch
import numpy as np
import sys
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cnn" / "src"))
from train_2d import build_model

images_dir = Path("../../cnn/models/alpha/predictions")
output_dir = Path("../../gradcam/results")
weights = Path("../../cnn/models/alpha/alpha.pth")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

checkpoint = torch.load(weights, map_location=device)
config = checkpoint["config"]
config["score_threshold"] = 0.6

model = build_model(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()
print("Model loaded")

# Last conv block of the ResNet-50 backbone — richest spatial features
target_layers = [model.backbone.body.layer4]

# EigenCAM works better than vanilla GradCAM for detection models —
# it doesn't need a classification target, just the feature activations
cam = EigenCAM(model=model, target_layers=target_layers, reshape_transform=None)

images = sorted(images_dir.glob("*.png"))
for i, img_path in enumerate(images):
    print(f"[{i+1}/{len(images)}] {img_path.name}")
    image = Image.open(img_path).convert("RGB")
    rgb_float = np.array(image).astype(np.float32) / 255.0
    tensor = T.functional.to_tensor(image).to(device).unsqueeze(0)

    # Run detection first to get the boxes
    with torch.no_grad():
        preds = model([tensor[0]])[0]
    keep = preds["scores"] >= config["score_threshold"]
    boxes = preds["boxes"][keep].cpu().numpy()
    scores = preds["scores"][keep].cpu().numpy()
    labels = preds["labels"][keep].cpu().numpy()

    if len(boxes) == 0:
        print("  no detections, skipping")
        continue

    # Build per-box targets so the CAM aggregates attention from every fault
    targets = [
        FasterRCNNBoxScoreTarget(
            labels=labels.tolist(),
            bounding_boxes=boxes.tolist(),
        )
    ]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(overlay)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
        )
        ax.text(
            x1,
            y1 - 5,
            f"{score:.2f}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="cyan"),
        )
    ax.set_title(f"Grad-CAM (EigenCAM) — {img_path.name}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / img_path.name, dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nDone. Heatmaps saved to {output_dir}")
