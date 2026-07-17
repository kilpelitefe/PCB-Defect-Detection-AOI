# PCB Otomatik Optik Muayene (AOI)

YOLOv8 ile PCB (baskılı devre kartı) üzerindeki üretim hatalarını tespit eden ve
Pix2Pix ile hatalı bölgeler için görsel onarım önerisi sunan bir Streamlit uygulaması.

## Özellikler

- **YOLOv8 hata tespiti**: PCB görüntüsündeki hataları tespit eder ve sınıflandırır
  (open, short, mousebite, spur, copper, pin-hole).
- **Otomatik siyah-beyaz dönüşümü**: Yüklenen renkli fotoğraflar YOLO'ya verilmeden
  önce gri tonlamaya çevrilir. Model DeepPCB'nin siyah-beyaz görüntüleriyle eğitildiği
  için tespit bu siyah-beyaz sürüm üzerinde yapılır.
- **Pix2Pix lokal onarım önerisi**: Tespit edilen hatalı bölgeler için tahmini temiz
  görünüm üretir (yalnızca DeepPCB benzeri siyah-beyaz görüntülerde anlamlıdır).
- **Model doğruluğu ölçümü**: Test seti üzerinde mAP, precision ve recall hesaplar.

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

## Model Dosyaları

Büyük model dosyaları GitHub'ın 100MB dosya sınırını aştığı için depoya dahil
**edilmemiştir**. Uygulamayı tam çalıştırmak için:

- **`best.pt`** — YOLOv8 hata tespit modeli (depoda mevcut).
- **`trained_pix2pix_model/`** — Pix2Pix generator ağırlıkları. Uygulama şu dosyalardan
  birini arar: `latest_net_G.pth`, `100_net_G.pth` veya `95_net_G.pth`. Bu dosyaları
  bu klasöre ayrıca eklemeniz gerekir. Bu dosyalar olmadan YOLO tespiti çalışır ancak
  Pix2Pix onarım modülü devre dışı kalır.

## Bağımlılık: pytorch-CycleGAN-and-pix2pix

Pix2Pix generator mimarisi (`define_G`) için
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
deposu kullanılır (BSD lisansı).

## Veri Seti

Model, [DeepPCB](https://github.com/tangsanli5201/DeepPCB) veri seti kullanılarak
eğitilmiştir.
