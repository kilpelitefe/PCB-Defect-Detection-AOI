import os
import cv2
import random
from glob import glob
from tqdm import tqdm

# Builds a full-image Pix2Pix (AB) dataset from DeepPCB:
# each output image is a 256x256 defective (_test) image and its clean (_temp)
# counterpart concatenated side by side, as expected by pytorch-CycleGAN-and-pix2pix.

source_root = "PCBData"
output_root = "pix2pix_pcb"

train_dir = os.path.join(output_root, "train")
test_dir = os.path.join(output_root, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

test_images = []

for ext in ["jpg", "jpeg", "png", "bmp"]:
    test_images.extend(glob(os.path.join(source_root, "**", f"*_test.{ext}"), recursive=True))

pairs = []

for test_path in test_images:
    temp_path = (
        test_path
        .replace("_test.jpg", "_temp.jpg")
        .replace("_test.jpeg", "_temp.jpeg")
        .replace("_test.png", "_temp.png")
        .replace("_test.bmp", "_temp.bmp")
    )

    if os.path.exists(temp_path):
        pairs.append((test_path, temp_path))

print("Matched pairs found:", len(pairs))

random.shuffle(pairs)

split = int(len(pairs) * 0.8)
train_pairs = pairs[:split]
test_pairs = pairs[split:]

def create_pair(test_path, temp_path, save_dir):
    test_img = cv2.imread(test_path)
    temp_img = cv2.imread(temp_path)

    if test_img is None or temp_img is None:
        return

    test_img = cv2.resize(test_img, (256, 256))
    temp_img = cv2.resize(temp_img, (256, 256))

    combined = cv2.hconcat([test_img, temp_img])

    name = os.path.basename(test_path).replace("_test.jpg", ".jpg").replace("_test.png", ".jpg").replace("_test.bmp", ".jpg")
    cv2.imwrite(os.path.join(save_dir, name), combined)

for test_path, temp_path in tqdm(train_pairs):
    create_pair(test_path, temp_path, train_dir)

for test_path, temp_path in tqdm(test_pairs):
    create_pair(test_path, temp_path, test_dir)

print("Pix2Pix dataset ready.")
