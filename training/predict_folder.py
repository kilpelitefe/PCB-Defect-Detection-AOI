from ultralytics import YOLO

# Quick sanity check: run the trained model on a folder of DeepPCB images
# and save the annotated results (runs/detect/predict*).

model = YOLO("best.pt")

results = model.predict(
    source="PCBData/group00041/00041",
    conf=0.25,
    save=True
)

print("Prediction finished.")
