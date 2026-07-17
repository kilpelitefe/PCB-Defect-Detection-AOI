import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFilter
from collections import Counter
import tempfile
import os
import cv2
import io
import sys
import torch
import torchvision.transforms as transforms
import numpy as np



PIX2PIX_REPO_PATH = "pytorch-CycleGAN-and-pix2pix"
sys.path.append(PIX2PIX_REPO_PATH)

from models.networks import define_G



st.set_page_config(
    page_title="PCB Defect Detection System",
    layout="wide"
)

st.title("Automated Optical Inspection")
st.write(
    "This system detects manufacturing defects on PCBs using a YOLOv8 model, "
    "shows the defect types, and provides visual repair suggestions with Pix2Pix "
    "for suitable image formats."
)



confidence_value = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

padding_value = st.sidebar.slider(
    "Pix2Pix box padding",
    min_value=5,
    max_value=80,
    value=25,
    step=5
)

st.sidebar.write("### Modules")
st.sidebar.write("- YOLOv8: Defect detection")
st.sidebar.write("- Pix2Pix: Visual repair suggestion")

st.sidebar.divider()
st.sidebar.write("### Pix2Pix Usage Note")
st.sidebar.info(
    "The Pix2Pix model produces meaningful results on DeepPCB-like black-and-white AOI images. "
    "For color photos or PCB images outside the dataset, Pix2Pix is disabled automatically."
)

force_pix2pix = st.sidebar.checkbox(
    "Force Pix2Pix even on out-of-dataset images",
    value=False
)

st.sidebar.divider()
st.sidebar.write("### Model Accuracy")
st.sidebar.caption("Measures the overall performance of the model on the test set.")

data_yaml_path = st.sidebar.text_input(
    "data.yaml path",
    value=r"C:\Users\kilpe\OneDrive\Masaüstü\AOI\yolo_dataset\data.yaml"
)



@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")


model = load_yolo_model()



if st.sidebar.button("Evaluate accuracy on test set"):
    if not os.path.exists(data_yaml_path):
        st.sidebar.error(f"data.yaml not found: {data_yaml_path}")
    else:
        with st.spinner("Evaluating on the test set. This may take a few minutes..."):
            metrics = model.val(
                data=data_yaml_path,
                split="test",
                verbose=False,
                plots=False,
            )

        st.session_state["test_metrics"] = {
            "map50": float(metrics.box.map50),
            "map": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

if "test_metrics" in st.session_state:
    m = st.session_state["test_metrics"]

    c1, c2 = st.sidebar.columns(2)
    c1.metric("mAP@0.5", f"{m['map50']:.3f}")
    c2.metric("mAP@0.5:0.95", f"{m['map']:.3f}")

    c3, c4 = st.sidebar.columns(2)
    c3.metric("Precision", f"{m['precision']:.3f}")
    c4.metric("Recall", f"{m['recall']:.3f}")

    st.sidebar.caption(
        "These scores are computed on test images the model has never seen during training. "
        "A high score indicates the model generalizes well on the DeepPCB test set."
    )



@st.cache_resource
def load_pix2pix_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=True,
        init_type="normal",
        init_gain=0.02
    )

    model_path = "trained_pix2pix_model/latest_net_G.pth"

    if not os.path.exists(model_path):
        model_path = "trained_pix2pix_model/100_net_G.pth"

    if not os.path.exists(model_path):
        model_path = "trained_pix2pix_model/95_net_G.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Pix2Pix generator model not found. "
            "The trained_pix2pix_model folder must contain latest_net_G.pth, 100_net_G.pth or 95_net_G.pth."
        )

    state_dict = torch.load(model_path, map_location=device)

    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata

    netG.load_state_dict(state_dict)
    netG.to(device)
    netG.eval()

    return netG, device, model_path


pix2pix_model, device, pix2pix_model_path = load_pix2pix_model()



def is_deeppcb_like(image: Image.Image, tolerance=8):

    img = np.array(image.convert("RGB")).astype(np.int16)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    rg_diff = np.abs(r - g)
    rb_diff = np.abs(r - b)
    gb_diff = np.abs(g - b)

    mean_diff = (rg_diff.mean() + rb_diff.mean() + gb_diff.mean()) / 3.0

    return mean_diff < tolerance, mean_diff


def run_pix2pix(input_image, apply_threshold=True, threshold=128):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_b = pix2pix_model(input_tensor)

    fake_b = fake_b.squeeze(0).cpu()
    fake_b = (fake_b * 0.5) + 0.5
    fake_b = torch.clamp(fake_b, 0, 1)

    fake_b_np = fake_b.permute(1, 2, 0).numpy()
    fake_b_np = (fake_b_np * 255).astype(np.uint8)

    result = Image.fromarray(fake_b_np)

    if apply_threshold:
        gray = result.convert("L")
        bw = gray.point(lambda p: 255 if p > threshold else 0)
        result = bw.convert("RGB")

    return result



def pix2pix_repair_full_image(original_image, boxes, padding=25, feather=8):
    orig_w, orig_h = original_image.size

    repaired_256 = run_pix2pix(original_image)
    repaired_full = repaired_256.resize((orig_w, orig_h), Image.LANCZOS)

    mask = Image.new("L", (orig_w, orig_h), 0)
    draw = ImageDraw.Draw(mask)

    repaired_patches = []

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(orig_w, x2 + padding)
        py2 = min(orig_h, y2 + padding)

        draw.rectangle([px1, py1, px2, py2], fill=255)

        original_crop = original_image.crop((px1, py1, px2, py2))
        repaired_crop = repaired_full.crop((px1, py1, px2, py2))

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        repaired_patches.append({
            "index": i,
            "class_name": class_name,
            "confidence": confidence,
            "box": (px1, py1, px2, py2),
            "original_crop": original_crop,
            "repaired_crop": repaired_crop,
        })

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    final_image = Image.composite(repaired_full, original_image, mask)

    return final_image, repaired_patches



fix_suggestions = {
    "open": "There is a break in the circuit trace. The broken copper trace can be repaired with solder or a jumper wire.",
    "short": "There is a short circuit between two conductive traces. Excess solder or copper should be cleaned.",
    "mousebite": "There is material loss on the copper trace. The area can be reinforced with solder.",
    "spur": "There is an unwanted copper protrusion. The excess copper should be scraped off or cleaned.",
    "copper": "There is a copper surface defect. Missing or excess copper areas should be inspected.",
    "pin-hole": "There is a small hole in the copper area. The hole can be filled with solder or copper repair material."
}

defect_descriptions = {
    "open": "A break in the copper trace or an interrupted circuit.",
    "short": "Two separate conductive traces unintentionally joined.",
    "mousebite": "A nibbled-looking missing region on the copper trace.",
    "spur": "An unwanted protrusion on the copper trace.",
    "copper": "Missing or excess material on the copper area.",
    "pin-hole": "A small hole on the copper surface."
}



uploaded_file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Convert the uploaded color image to black-and-white (grayscale).
    # The YOLO model was trained on DeepPCB's black-and-white images, so
    # defect detection always runs on this grayscale version.
    # convert("L") produces a single-channel grayscale image; convert("RGB")
    # brings it back to 3 channels (YOLO and visualization expect 3 channels).
    gray_image = image.convert("L").convert("RGB")

    deeppcb_like, color_difference = is_deeppcb_like(image)

    if deeppcb_like:
        st.success(
            "This image looks close to the DeepPCB-like black-and-white AOI format. "
            "The Pix2Pix visual repair suggestion can be enabled."
        )
        use_pix2pix = True
    else:
        st.warning(
            "This image does not look like the DeepPCB black-and-white AOI format. "
            "YOLO prediction can still run; however, the Pix2Pix model only produces "
            "meaningful results on DeepPCB-like images, so Pix2Pix has been disabled "
            "automatically for this image."
        )
        use_pix2pix = False

    if force_pix2pix:
        st.info(
            "Forcing Pix2Pix was selected in the sidebar. "
            "In this case the results should be considered experimental."
        )
        use_pix2pix = True

    st.caption(f"Image color difference score: {color_difference:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Black & White (YOLO Input)")
        st.image(gray_image, use_container_width=True)
        st.caption("YOLO detection runs on this black-and-white image.")

    # Feed YOLO the black-and-white version instead of the original color image.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        gray_image.save(tmp.name)
        image_path = tmp.name

    if st.button("Detect Defects"):
        results = model.predict(
            source=image_path,
            conf=confidence_value,
            save=False
        )

        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.subheader("YOLO Detection Result")
        st.image(result_img_rgb, use_container_width=True)

        boxes = results[0].boxes

        st.divider()

        if len(boxes) == 0:
            st.success("No defects were detected.")
        else:
            detected_classes = []

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)

            counter = Counter(detected_classes)
            total_errors = sum(counter.values())
            most_common_error = counter.most_common(1)[0][0]

            st.subheader("Overall Summary")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric("Total Defects", total_errors)

            with metric2:
                st.metric("Distinct Defect Types", len(counter))

            with metric3:
                st.metric("Most Common Defect", most_common_error)

            if total_errors <= 2:
                st.info("Overall status: A small number of defects were detected. The board should be inspected.")
            elif total_errors <= 6:
                st.warning("Overall status: A moderate number of defects were detected. Manual inspection is recommended.")
            else:
                st.error("Overall status: A large number of defects were detected. The board is at risk.")

            avg_confidence = sum(float(b.conf[0]) for b in boxes) / len(boxes)
            st.metric("Average Detection Confidence", f"{avg_confidence * 100:.1f}%")

            if avg_confidence < 0.45:
                st.warning(
                    "Average confidence is low. This image may differ from the DeepPCB images the model was trained on. "
                    "The results should be considered experimental."
                )

            st.subheader("Defect Counter")

            count_cols = st.columns(3)

            for index, (defect, count) in enumerate(counter.items()):
                with count_cols[index % 3]:
                    st.metric(defect, count)

            st.subheader("Detailed Defect List")

            for i, box in enumerate(boxes, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                with st.expander(f"Defect {i}: {class_name} — {confidence * 100:.2f}% confidence"):
                    st.write("**Defect description:**")
                    st.write(defect_descriptions.get(class_name, "No description found."))

                    st.write("**Suggested fix:**")
                    st.info(fix_suggestions.get(class_name, "No suggestion found for this defect."))

                    st.progress(confidence)

            st.divider()
            st.subheader("Pix2Pix Local Repair Module")

            if use_pix2pix:
                st.info(
                    "The Pix2Pix model generates an estimated clean appearance for the image. "
                    "The output is applied as a visual repair suggestion only on the defective regions detected by YOLO."
                )

                locally_repaired_image, repaired_patches = pix2pix_repair_full_image(
                    original_image=image,
                    boxes=boxes,
                    padding=padding_value
                )

                pix_col1, pix_col2 = st.columns(2)

                with pix_col1:
                    st.write("YOLO Defect Detection")
                    st.image(result_img_rgb, use_container_width=True)

                with pix_col2:
                    st.write("Pix2Pix Local Repair Suggestion")
                    st.image(locally_repaired_image, use_container_width=True)

                repaired_buffer = io.BytesIO()
                locally_repaired_image.save(repaired_buffer, format="PNG")

                st.download_button(
                    label="Download Local Repair Suggestion Image",
                    data=repaired_buffer.getvalue(),
                    file_name="pcb_local_pix2pix_repaired.png",
                    mime="image/png"
                )

                st.subheader("Defect Regions Before / After")

                st.caption(
                    "Each row shows a defect region found by YOLO and the local repair "
                    "suggestion generated by Pix2Pix for that region."
                )

                for patch in repaired_patches:
                    with st.expander(
                        f"Defect {patch['index']}: {patch['class_name']} — {patch['confidence'] * 100:.2f}% confidence"
                    ):
                        before_col, after_col = st.columns(2)

                        with before_col:
                            st.write("Defective Region")
                            st.image(patch["original_crop"], use_container_width=True)

                        with after_col:
                            st.write("Pix2Pix Local Repair Suggestion")
                            st.image(patch["repaired_crop"], use_container_width=True)
            else:
                st.warning(
                    "Pix2Pix was not run for this image. "
                    "Reason: the image does not resemble the DeepPCB dataset format. "
                    "The Pix2Pix model does not produce reliable results on such color "
                    "or out-of-dataset PCB photos."
                )

            st.divider()
            st.subheader("Defect Types and Fixes")

            table_data = []

            for defect in counter.keys():
                table_data.append(
                    {
                        "Defect Type": defect,
                        "Meaning": defect_descriptions.get(defect, "-"),
                        "Fix": fix_suggestions.get(defect, "-")
                    }
                )

            st.table(table_data)

    os.remove(image_path)

else:
    st.info("Please upload a PCB image to analyze.")
