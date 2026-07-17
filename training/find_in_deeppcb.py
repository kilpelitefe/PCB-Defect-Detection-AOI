from pathlib import Path
from PIL import Image
import numpy as np

# Utility: checks whether a given image comes from (or is derived from) the
# DeepPCB dataset by finding the most similar image via mean pixel difference.

# Image to check — adjust to your own path
TARGET_IMAGE = "my_image.jpg"

# DeepPCB dataset root — adjust to your local path if needed
PCBDATA_ROOT = Path("PCBData")

target = Image.open(TARGET_IMAGE).convert("L")
target = target.resize((640, 640))
target_arr = np.array(target)

closest = None
lowest_diff = float("inf")

for jpg in PCBDATA_ROOT.rglob("*.jpg"):
    try:
        img = Image.open(jpg).convert("L").resize((640, 640))
        diff = np.mean(np.abs(np.array(img).astype(int) - target_arr.astype(int)))
        if diff < lowest_diff:
            lowest_diff = diff
            closest = jpg.name
    except Exception:
        continue

print(f"Most similar file: {closest}")
print(f"Mean pixel difference: {lowest_diff:.2f}")
print()
if lowest_diff < 5:
    print(">>> This image IS in DeepPCB (near-exact match)")
elif lowest_diff < 25:
    print(">>> Very similar, probably a processed version of a DeepPCB image")
else:
    print(">>> Does NOT appear to be in DeepPCB (external image)")
