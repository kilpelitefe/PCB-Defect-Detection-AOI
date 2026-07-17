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
    page_title="PCB Hata Tespit Sistemi",
    layout="wide"
)

st.title("Otomatik Optik Muayene")
st.write(
    "Bu sistem YOLOv8 modeli ile PCB üzerindeki üretim hatalarını tespit eder, "
    "hata türlerini gösterir ve uygun görüntü formatlarında Pix2Pix ile görsel onarım önerisi sunar."
)



confidence_value = st.sidebar.slider(
    "Güven eşiği",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

padding_value = st.sidebar.slider(
    "Pix2Pix kutu genişletme payı",
    min_value=5,
    max_value=80,
    value=25,
    step=5
)

st.sidebar.write("### Modüller")
st.sidebar.write("- YOLOv8: Hata tespiti")
st.sidebar.write("- Pix2Pix: Görsel onarım önerisi")

st.sidebar.divider()
st.sidebar.write("### Pix2Pix Kullanım Notu")
st.sidebar.info(
    "Pix2Pix modeli DeepPCB benzeri siyah-beyaz AOI görüntülerinde anlamlı sonuç üretir. "
    "Renkli veya veri seti dışı PCB fotoğraflarında Pix2Pix otomatik olarak devre dışı bırakılır."
)

force_pix2pix = st.sidebar.checkbox(
    "Pix2Pix'i veri seti dışı görüntülerde de zorla çalıştır",
    value=False
)

st.sidebar.divider()
st.sidebar.write("### Model Doğruluğu")
st.sidebar.caption("Test seti üzerinde modelin genel başarısını ölçer.")

data_yaml_path = st.sidebar.text_input(
    "data.yaml yolu",
    value=r"C:\Users\kilpe\OneDrive\Masaüstü\AOI\yolo_dataset\data.yaml"
)



@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")


model = load_yolo_model()



if st.sidebar.button("Test setinde doğruluğu ölç"):
    if not os.path.exists(data_yaml_path):
        st.sidebar.error(f"data.yaml bulunamadı: {data_yaml_path}")
    else:
        with st.spinner("Test seti değerlendiriliyor. Bu işlem birkaç dakika sürebilir..."):
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
        "Bu skorlar modelin eğitimde görmediği test görüntülerinden hesaplanır. "
        "Yüksek skor, modelin DeepPCB test setinde başarılı genelleme yaptığını gösterir."
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
            "Pix2Pix generator modeli bulunamadı. "
            "trained_pix2pix_model klasöründe latest_net_G.pth, 100_net_G.pth veya 95_net_G.pth olmalı."
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
    "open": "Devre hattında kopukluk vardır. Kopan bakır yol lehim veya jumper kablo ile onarılabilir.",
    "short": "İki iletken yol arasında kısa devre vardır. Fazla lehim veya bakır temizlenmelidir.",
    "mousebite": "Bakır hatta eksilme vardır. Bölge lehimle güçlendirilebilir.",
    "spur": "İstenmeyen bakır çıkıntısı vardır. Fazlalık bakır kazınmalı veya temizlenmelidir.",
    "copper": "Bakır yüzey hatası vardır. Eksik veya fazla bakır alan kontrol edilmelidir.",
    "pin-hole": "Bakır bölgede küçük delik vardır. Delik lehim veya bakır tamir malzemesiyle kapatılabilir."
}

defect_descriptions = {
    "open": "Bakır yolun kopması veya devrenin kesilmesi.",
    "short": "İki farklı iletken yolun istenmeden birleşmesi.",
    "mousebite": "Bakır hatta kemirilmiş gibi eksik bölge oluşması.",
    "spur": "Bakır yolda istenmeyen çıkıntı oluşması.",
    "copper": "Bakır alanında eksiklik veya fazlalık bulunması.",
    "pin-hole": "Bakır yüzey üzerinde küçük delik oluşması."
}



uploaded_file = st.file_uploader(
    "PCB görüntüsü yükle",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Yüklenen renkli görüntüyü siyah-beyaz (gri tonlama) hale getir.
    # YOLO modeli DeepPCB'nin siyah-beyaz görüntüleri üzerinde eğitildiği için
    # hata tespiti her zaman bu siyah-beyaz sürüm üzerinde yapılır.
    # convert("L") tek kanallı gri görüntü üretir, convert("RGB") ile tekrar
    # 3 kanala çevrilir (YOLO ve görselleştirme 3 kanal bekler).
    gray_image = image.convert("L").convert("RGB")

    deeppcb_like, color_difference = is_deeppcb_like(image)

    if deeppcb_like:
        st.success(
            "Bu görüntü DeepPCB benzeri siyah-beyaz AOI formatına yakın görünüyor. "
            "Pix2Pix görsel onarım önerisi etkinleştirilebilir."
        )
        use_pix2pix = True
    else:
        st.warning(
            "Bu görüntü DeepPCB benzeri siyah-beyaz AOI formatında görünmüyor. "
            "YOLO tahmini çalıştırılabilir; ancak Pix2Pix modeli yalnızca DeepPCB benzeri görüntülerde "
            "anlamlı sonuç üretir. Bu nedenle Pix2Pix bu görüntüde otomatik olarak devre dışı bırakıldı."
        )
        use_pix2pix = False

    if force_pix2pix:
        st.info(
            "Sidebar üzerinden Pix2Pix çalıştırma seçildi. "
            "Bu durumda sonuçlar deneysel olarak değerlendirilmelidir."
        )
        use_pix2pix = True

    st.caption(f"Görüntü renk farkı skoru: {color_difference:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yüklenen Görüntü")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Siyah-Beyaz (YOLO Girişi)")
        st.image(gray_image, use_container_width=True)
        st.caption("YOLO tespiti bu siyah-beyaz görüntü üzerinde yapılır.")

    # YOLO'ya orijinal renkli görüntü yerine siyah-beyaz sürümü ver.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        gray_image.save(tmp.name)
        image_path = tmp.name

    if st.button("Hata Tespit Et"):
        results = model.predict(
            source=image_path,
            conf=confidence_value,
            save=False
        )

        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.subheader("YOLO Tespit Sonucu")
        st.image(result_img_rgb, use_container_width=True)

        boxes = results[0].boxes

        st.divider()

        if len(boxes) == 0:
            st.success("Herhangi bir hata tespit edilmedi.")
        else:
            detected_classes = []

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)

            counter = Counter(detected_classes)
            total_errors = sum(counter.values())
            most_common_error = counter.most_common(1)[0][0]

            st.subheader("Genel Durum Özeti")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric("Toplam Hata", total_errors)

            with metric2:
                st.metric("Farklı Hata Türü", len(counter))

            with metric3:
                st.metric("En Çok Görülen Hata", most_common_error)

            if total_errors <= 2:
                st.info("Genel durum: Az sayıda hata tespit edildi. Kart kontrol edilmelidir.")
            elif total_errors <= 6:
                st.warning("Genel durum: Orta seviyede hata tespit edildi. Manuel inceleme önerilir.")
            else:
                st.error("Genel durum: Çok sayıda hata tespit edildi. Kart risklidir.")

            avg_confidence = sum(float(b.conf[0]) for b in boxes) / len(boxes)
            st.metric("Ortalama Tespit Güveni", f"%{avg_confidence * 100:.1f}")

            if avg_confidence < 0.45:
                st.warning(
                    "Ortalama güven düşük. Bu görüntü modelin eğitildiği DeepPCB görüntülerinden farklı olabilir. "
                    "Sonuçlar deneysel olarak değerlendirilmelidir."
                )

            st.subheader("Hata Sayacı")

            count_cols = st.columns(3)

            for index, (defect, count) in enumerate(counter.items()):
                with count_cols[index % 3]:
                    st.metric(defect, count)

            st.subheader("Detaylı Hata Listesi")

            for i, box in enumerate(boxes, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                with st.expander(f"{i}. Hata: {class_name} — %{confidence * 100:.2f} güven"):
                    st.write("**Hata açıklaması:**")
                    st.write(defect_descriptions.get(class_name, "Açıklama bulunamadı."))

                    st.write("**Çözüm önerisi:**")
                    st.info(fix_suggestions.get(class_name, "Bu hata için öneri bulunamadı."))

                    st.progress(confidence)

            st.divider()
            st.subheader("Pix2Pix Lokal Onarım Modülü")

            if use_pix2pix:
                st.info(
                    "Pix2Pix modeli görüntü için tahmini temiz görünüm üretir. "
                    "Üretilen çıktı yalnızca YOLO'nun tespit ettiği hatalı bölgelerde görsel onarım önerisi olarak uygulanır."
                )

                locally_repaired_image, repaired_patches = pix2pix_repair_full_image(
                    original_image=image,
                    boxes=boxes,
                    padding=padding_value
                )

                pix_col1, pix_col2 = st.columns(2)

                with pix_col1:
                    st.write("YOLO Hata Tespiti")
                    st.image(result_img_rgb, use_container_width=True)

                with pix_col2:
                    st.write("Pix2Pix Lokal Onarım Önerisi")
                    st.image(locally_repaired_image, use_container_width=True)

                repaired_buffer = io.BytesIO()
                locally_repaired_image.save(repaired_buffer, format="PNG")

                st.download_button(
                    label="Lokal Onarım Önerisi Görselini İndir",
                    data=repaired_buffer.getvalue(),
                    file_name="pcb_local_pix2pix_repaired.png",
                    mime="image/png"
                )

                st.subheader("Hata Bölgeleri Önce / Sonra")

                st.caption(
                    "Her satırda YOLO'nun bulduğu hata bölgesi ve Pix2Pix'in o bölge için ürettiği "
                    "lokal onarım önerisi gösterilir."
                )

                for patch in repaired_patches:
                    with st.expander(
                        f"{patch['index']}. Hata: {patch['class_name']} — %{patch['confidence'] * 100:.2f} güven"
                    ):
                        before_col, after_col = st.columns(2)

                        with before_col:
                            st.write("Hatalı Bölge")
                            st.image(patch["original_crop"], use_container_width=True)

                        with after_col:
                            st.write("Pix2Pix Lokal Onarım Önerisi")
                            st.image(patch["repaired_crop"], use_container_width=True)
            else:
                st.warning(
                    "Pix2Pix bu görüntü için çalıştırılmadı. "
                    "Sebep: Görüntü DeepPCB veri seti formatına benzemiyor. "
                    "Pix2Pix modeli bu tür renkli veya veri seti dışı PCB fotoğraflarında güvenilir sonuç üretmez."
                )

            st.divider()
            st.subheader("Hata Türleri ve Çözüm Tablosu")

            table_data = []

            for defect in counter.keys():
                table_data.append(
                    {
                        "Hata Türü": defect,
                        "Anlamı": defect_descriptions.get(defect, "-"),
                        "Çözüm": fix_suggestions.get(defect, "-")
                    }
                )

            st.table(table_data)

    os.remove(image_path)

else:
    st.info("Lütfen analiz etmek için bir PCB görüntüsü yükleyin.")