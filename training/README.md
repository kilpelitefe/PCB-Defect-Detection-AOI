# Model Training

Scripts used to prepare the datasets and train the two models used by the app:
the YOLOv8 defect detector (`best.pt`) and the Pix2Pix repair generator
(`trained_pix2pix_model/*_net_G.pth`).

## Prerequisites

- The [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset, extracted so
  that a `PCBData/` folder sits next to these scripts (or adjust the path
  variables at the top of each script).
- Extra dependency for the dataset scripts: `pip install tqdm`
- For Pix2Pix training: the
  [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  repository (already included in this repo).

## Pipeline

### 0. (Optional) Verify the dataset

```bash
python check_dataset.py
```

Walks `PCBData/` and reports missing `_test`/`_temp` pairs, identical pairs and
size mismatches.

### 1. Convert DeepPCB to YOLOv8 format

```bash
python prepare_yolo_dataset.py --source PCBData --output yolo_dataset
```

Maps DeepPCB classes (1-6) to YOLO ids (0-5: open, short, mousebite, spur,
copper, pin-hole), converts corner-format boxes to normalized YOLO format,
splits into train/val/test (70/20/10 by default) and writes `data.yaml`.

### 2. Train YOLOv8

```bash
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Copy the resulting `runs/detect/train/weights/best.pt` to the repo root for the
app to use.

### 3. Build the Pix2Pix dataset

Two variants were experimented with:

```bash
# Full-image pairs (256x256, defective|clean side by side)
python prepare_pix2pix_dataset.py

# Crop-based pairs (128x128 defect regions from YOLO labels)
python prepare_pix2pix_crops.py
```

Both produce AB-format images (input and target concatenated horizontally) as
expected by pytorch-CycleGAN-and-pix2pix.

### 4. Train Pix2Pix

```bash
cd pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ../pix2pix_pcb --name pcb_pix2pix --model pix2pix --direction AtoB
```

Copy the trained generator checkpoints (e.g. `latest_net_G.pth`) into
`trained_pix2pix_model/` for the app to use.

## Utilities

- `predict_folder.py` — runs `best.pt` on a folder of images and saves
  annotated results (quick sanity check).
- `find_in_deeppcb.py` — checks whether a given image originates from the
  DeepPCB dataset by nearest mean-pixel-difference search.
