import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

# Builds a crop-based Pix2Pix (AB) dataset: for every YOLO-labeled defect box,
# crops the defective region from the _test image and the same region from the
# clean _temp image, then concatenates them side by side.

# Root folders — adjust to your local paths if needed
YOLO_ROOT = Path("yolo_dataset")
PCB_ROOT = Path("PCBData")
OUTPUT_ROOT = Path("pix2pix_crop_pcb")

# Settings
CROP_SIZE = 128
PADDING = 25

# Output folders
for split in ["train", "test"]:
    (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(label_line, img_w, img_h, padding=25):
    """
    YOLO format:
    class_id x_center y_center width height
    values are in the 0-1 range.
    """
    parts = label_line.strip().split()

    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    box_w = float(parts[3]) * img_w
    box_h = float(parts[4]) * img_h

    x1 = int(x_center - box_w / 2) - padding
    y1 = int(y_center - box_h / 2) - padding
    x2 = int(x_center + box_w / 2) + padding
    y2 = int(y_center + box_h / 2) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return class_id, x1, y1, x2, y2


def find_temp_image(test_image_path):
    """
    Finds the corresponding _temp image in PCBData from a YOLO image name.
    Example:
    00041000_test.jpg -> 00041000_temp.jpg
    """
    test_name = test_image_path.name

    if "_test" not in test_name:
        return None

    temp_name = test_name.replace("_test", "_temp")

    matches = list(PCB_ROOT.rglob(temp_name))

    if len(matches) == 0:
        return None

    return matches[0]


def create_pair_crop(test_img, temp_img, x1, y1, x2, y2):
    defective_crop = test_img[y1:y2, x1:x2]
    clean_crop = temp_img[y1:y2, x1:x2]

    if defective_crop.size == 0 or clean_crop.size == 0:
        return None

    defective_crop = cv2.resize(defective_crop, (CROP_SIZE, CROP_SIZE))
    clean_crop = cv2.resize(clean_crop, (CROP_SIZE, CROP_SIZE))

    combined = cv2.hconcat([defective_crop, clean_crop])

    return combined


all_pairs = []

for split in ["train", "val", "test"]:
    image_dir = YOLO_ROOT / "images" / split
    label_dir = YOLO_ROOT / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        continue

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(image_dir.glob(ext))

    for image_path in tqdm(image_paths, desc=f"scanning {split}"):
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        temp_path = find_temp_image(image_path)

        if temp_path is None:
            continue

        test_img = cv2.imread(str(image_path))
        temp_img = cv2.imread(str(temp_path))

        if test_img is None or temp_img is None:
            continue

        img_h, img_w = test_img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            box = yolo_to_xyxy(line, img_w, img_h, padding=PADDING)

            if box is None:
                continue

            class_id, x1, y1, x2, y2 = box

            pair_crop = create_pair_crop(test_img, temp_img, x1, y1, x2, y2)

            if pair_crop is None:
                continue

            all_pairs.append((split, image_path.stem, idx, class_id, pair_crop))


print("Total crop pairs:", len(all_pairs))

random.shuffle(all_pairs)

split_index = int(len(all_pairs) * 0.8)
train_pairs = all_pairs[:split_index]
test_pairs = all_pairs[split_index:]


def save_pairs(pairs, output_split):
    for i, (_, stem, idx, class_id, pair_crop) in enumerate(tqdm(pairs, desc=f"saving {output_split}")):
        save_name = f"{stem}_box{idx}_class{class_id}_{i}.jpg"
        save_path = OUTPUT_ROOT / output_split / save_name
        cv2.imwrite(str(save_path), pair_crop)


save_pairs(train_pairs, "train")
save_pairs(test_pairs, "test")

print("Crop-based Pix2Pix dataset ready.")
print("Train:", len(list((OUTPUT_ROOT / "train").glob('*.jpg'))))
print("Test:", len(list((OUTPUT_ROOT / "test").glob('*.jpg'))))
