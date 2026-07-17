import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


# DeepPCB class ids: 1=open, 2=short, 3=mousebite, 4=spur, 5=copper, 6=pin-hole
CLASS_MAPPING = {
    1: 0,  # open
    2: 1,  # short
    3: 2,  # mousebite
    4: 3,  # spur
    5: 4,  # copper
    6: 5,  # pin-hole
}

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def convert_bbox_to_yolo(x1: float, y1: float, x2: float, y2: float,
                          img_w: int, img_h: int) -> tuple[float, float, float, float]:

    # Some annotations may have x1 > x2, normalize to be safe
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    # Clamp to the 0-1 range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def convert_annotation(txt_path: Path, img_path: Path, output_txt: Path) -> int:

    # Read the actual size from the image (measured value is safer than assumption)
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    yolo_lines = []
    skipped = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"  {txt_path.name}:{line_num} -> expected 5 values, found {len(parts)}, skipped")
                skipped += 1
                continue

            try:
                x1, y1, x2, y2 = map(float, parts[:4])
                deeppcb_class = int(parts[4])
            except ValueError:
                print(f"  {txt_path.name}:{line_num} -> numeric conversion error, skipped")
                skipped += 1
                continue

            if deeppcb_class not in CLASS_MAPPING:
                print(f"  {txt_path.name}:{line_num} -> unknown class {deeppcb_class}, skipped")
                skipped += 1
                continue

            yolo_class = CLASS_MAPPING[deeppcb_class]
            xc, yc, w, h = convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)

            if w <= 0 or h <= 0:
                print(f"  {txt_path.name}:{line_num} -> invalid box size, skipped")
                skipped += 1
                continue

            yolo_lines.append(f"{yolo_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    return len(yolo_lines)


def find_image_annotation_pairs(source_dir: Path) -> list[tuple[Path, Path]]:

    pairs = []
    skipped_no_image = 0


    for txt_file in source_dir.rglob("*.txt"):

        if txt_file.parent == source_dir:
            continue
        if txt_file.stem in ("trainval", "test", "train", "val"):
            continue


        ann_folder = txt_file.parent
        if not ann_folder.name.endswith("_not"):

            img_file = txt_file.with_name(f"{txt_file.stem}_test.jpg")
            if img_file.exists():
                pairs.append((img_file, txt_file))
            continue


        img_folder_name = ann_folder.name[:-4]
        img_folder = ann_folder.parent / img_folder_name

        if not img_folder.exists():
            skipped_no_image += 1
            continue


        img_file = img_folder / f"{txt_file.stem}_test.jpg"

        if img_file.exists():
            pairs.append((img_file, txt_file))
        else:
            skipped_no_image += 1

    if skipped_no_image > 0:
        print(f"No matching image found for {skipped_no_image} annotations")

    return pairs


def split_dataset(pairs: list, train_ratio: float, val_ratio: float, seed: int = 42):

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def process_split(pairs: list, split_name: str, output_dir: Path) -> dict:

    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "annotations": 0, "empty": 0}

    print(f"\n[{split_name.upper()}] processing {len(pairs)} samples...")

    for img_path, txt_path in pairs:

        new_img_path = img_out / img_path.name
        shutil.copy2(img_path, new_img_path)


        new_lbl_path = lbl_out / f"{img_path.stem}.txt"

        n_anns = convert_annotation(txt_path, img_path, new_lbl_path)

        stats["images"] += 1
        stats["annotations"] += n_anns
        if n_anns == 0:
            stats["empty"] += 1

    return stats


def write_data_yaml(output_dir: Path):

    yaml_content = f"""# DeepPCB - YOLOv8 format


path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test


nc: {len(CLASS_NAMES)}



"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[OK] data.yaml created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="DeepPCB -> YOLOv8 format conversion")
    parser.add_argument("--source", type=str, required=True,
                        help="DeepPCB dataset root folder (contains PCBData)")
    parser.add_argument("--output", type=str, required=True,
                        help="Folder where the YOLO-format output will be written")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)



    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print("[ERROR] train_ratio + val_ratio cannot exceed 1.0")
        return


    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Split:  train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, test={test_ratio:.0%}")


    print("\n[1/4] Image-annotation pairs")
    pairs = find_image_annotation_pairs(source_dir)
    print(f"      {len(pairs)} pairs found")

    print("\n[2/4] Dataset split")
    train, val, test = split_dataset(pairs, args.train_ratio, args.val_ratio, args.seed)
    print(f"      train: {len(train)}, val: {len(val)}, test: {len(test)}")

    print("\n[3/4] Copying images and creating labels")
    all_stats = {}
    for split_name, split_pairs in [("train", train), ("val", val), ("test", test)]:
        all_stats[split_name] = process_split(split_pairs, split_name, output_dir)

    print("\n[4/4] Writing data.yaml")
    write_data_yaml(output_dir)



    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, stats in all_stats.items():
        print(f"  {split_name:6s}: {stats['images']:4d} images | "
              f"{stats['annotations']:5d} defects | "
              f"{stats['empty']:3d} empty labels")

    total_anns = sum(s["annotations"] for s in all_stats.values())
    print(f"\n  Total: {sum(s['images'] for s in all_stats.values())} images, {total_anns} defect annotations")
    print(f"\n[DONE] Ready for YOLOv8 training. Command:")
    print(f"  yolo detect train data={output_dir}/data.yaml model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--source", "PCBData",
            "--output", "yolo_dataset"
        ])
    main()
