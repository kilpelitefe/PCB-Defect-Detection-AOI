# PCB Automated Optical Inspection (AOI)

A Streamlit application that detects manufacturing defects on PCBs (printed circuit
boards) using YOLOv8 and provides visual repair suggestions for defective regions
using Pix2Pix.

## Features

- **YOLOv8 defect detection**: Detects and classifies defects on PCB images
  (open, short, mousebite, spur, copper, pin-hole).
- **Automatic grayscale conversion**: Uploaded color photos are converted to
  grayscale before being passed to YOLO. Since the model was trained on DeepPCB's
  black-and-white images, detection runs on this grayscale version.
- **Pix2Pix local repair suggestion**: Generates an estimated clean appearance for
  detected defective regions (only meaningful on DeepPCB-like black-and-white images).
- **Model accuracy evaluation**: Computes mAP, precision, and recall on the test set.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Model Files

Large model files are **not included** in this repository because they exceed
GitHub's 100MB file size limit. To run the full application:

- **`best.pt`** — YOLOv8 defect detection model (included in the repo).
- **`trained_pix2pix_model/`** — Pix2Pix generator weights. The application looks
  for one of the following files: `latest_net_G.pth`, `100_net_G.pth`, or
  `95_net_G.pth`. You need to add these files to this folder separately. Without
  them, YOLO detection still works but the Pix2Pix repair module is disabled.

## Dependency: pytorch-CycleGAN-and-pix2pix

The Pix2Pix generator architecture (`define_G`) is provided by the
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
repository (BSD license).

## Dataset

The model was trained on the [DeepPCB](https://github.com/tangsanli5201/DeepPCB)
dataset.
