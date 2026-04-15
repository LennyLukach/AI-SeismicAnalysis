import torch
import sys
from pathlib import Path
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_2d import build_model, visualize_predictions

images_dir = Path("../../pipeline/results/run_20260407_171623/images")
output_dir = Path("../models/alpha/predictions")
weights = Path("../models/alpha/alpha.pth")

output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

checkpoint = torch.load(weights, map_location=device)
config = checkpoint["config"]
config["score_threshold"] = 0.6 # this is the confidence the model needs to label a fault || modify this
model = build_model(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("Model loaded")

images = sorted(images_dir.glob("*.png"))

for i, img_path in enumerate(images):
    out_path = output_dir / img_path.name
    print(f"[{i+1}/{len(images)}] {img_path.name}")

    image = Image.open(img_path).convert("RGB")
    tensor = T.functional.to_tensor(image).to(device)

    with torch.no_grad():
        preds = model([tensor])[0]

    keep = preds["scores"] >= config["score_threshold"]
    boxes = preds["boxes"][keep].cpu().numpy()
    scores = preds["scores"][keep].cpu().numpy()

    visualize_predictions(image, boxes, scores, str(out_path))

print(f"\nDone. Results saved to {output_dir}")