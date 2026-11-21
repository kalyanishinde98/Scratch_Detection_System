
from pathlib import Path
from ultralytics import YOLO
import torch

# Project directories
ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "scratch_yolo" / "data.yaml"

# Check dataset configuration
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

print("=" * 50)
print("SMART SURFACE SCRATCH DETECTION")
print("YOLO11n MODEL TRAINING")
print("=" * 50)

# Hardware information
print("\nHardware:")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Training device: CPU")

# Load pretrained YOLO11 nano model
print("\nLoading YOLO11n pretrained model...")
model = YOLO("yolo11n.pt")

print("\nStarting initial training test...")

# Train model
results = model.train(
    data=str(DATA_YAML),
    epochs=2,
    imgsz=320,
    batch=1,
    workers=0,
    device="cpu",
    project=str(ROOT / "runs"),
    name="scratch_yolo11n_smoke",
    exist_ok=True,
    patience=5,
    plots=True,
    verbose=True
)

print("\n" + "=" * 50)
print("INITIAL TRAINING TEST FINISHED")
print("=" * 50)

print("\nTraining results saved in:")
print(ROOT / "runs" / "scratch_yolo11n_smoke")

print("\nBest model location:")
print(ROOT / "runs" / "scratch_yolo11n_smoke" / "weights" / "best.pt")
