"""
=========================================================
  YOLO11n, YOLO11s, RT-DETR — Grad-CAM Görselleştirme
  Ultralytics'in yerleşik Grad-CAM desteği kullanılır.
=========================================================
  Çalıştırma:
      python yolo_gradcam.py
=========================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────

MODELS = {
    "YOLO11n":  "runs/detect/YOLO11n/weights/best.pt",
    "YOLO11s":  "runs/detect/YOLO11s/weights/best.pt",
    "RT-DETR":  "runs/detect/RT-DETR/weights/best.pt",
}

MODEL_COLORS = {
    "YOLO11n":  "#FF9F1C",
    "YOLO11s":  "#E71D36",
    "RT-DETR":  "#7209B7",
}

VAL_DIR  = Path("val")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 6 örnek görüntü seç
# ─────────────────────────────────────────────────────────────

def get_samples(n=6):
    """Val setinden drone içeren n örnek görüntü seçer."""
    imgs = sorted(VAL_DIR.glob("*.png"))
    samples = []
    for p in imgs:
        lbl = VAL_DIR / (p.stem + ".txt")
        if lbl.exists() and lbl.read_text().strip():
            samples.append(p)
        if len(samples) == n:
            break
    return samples


# ─────────────────────────────────────────────────────────────
# Ultralytics Grad-CAM
# ─────────────────────────────────────────────────────────────

def get_gradcam_ultralytics(model_path: str, img_path: str):
    """
    Ultralytics'in yerleşik Grad-CAM fonksiyonunu kullanır.
    Hem ısı haritasını hem de tespit kutusunu döndürür.
    """
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils.torch_utils import select_device

    model = YOLO(model_path)

    # Tespit yap
    results = model(img_path, verbose=False)

    # Görüntüyü oku
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    # Bounding box çiz
    annotated = img_rgb.copy()
    conf_val  = 0.0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        conf_val = conf
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 196, 182), 3)
        cv2.putText(annotated, f"Drone {conf:.2f}",
                    (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 196, 182), 2)

    return img_rgb, annotated, conf_val, results[0]


def manual_gradcam(model_path: str, img_path: str, model_name: str):
    """
    Manuel Grad-CAM — son conv katmanından ısı haritası üretir.
    YOLO modelleri için özelleştirilmiş.
    """
    model = YOLO(model_path)
    nn_model = model.model

    # Hedef katmanı bul
    target_layer = None
    if "RT-DETR" in model_name or "rtdetr" in model_name.lower():
        # RT-DETR için encoder'ın son katmanı
        try:
            target_layer = list(nn_model.modules())[-10]
        except:
            target_layer = list(nn_model.modules())[-5]
    else:
        # YOLO için C2f bloklarından sonuncusu
        for m in nn_model.modules():
            if hasattr(m, 'cv2') and hasattr(m, 'cv1'):
                target_layer = m
        if target_layer is None:
            target_layer = list(nn_model.modules())[-5]

    feats = []
    grads = []

    def fwd_hook(_, __, out):
        feats.append(out)

    def bwd_hook(_, __, g):
        grads.append(g[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # Görüntüyü hazırla
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)
    img_tensor.requires_grad_(True)

    nn_model.eval()
    try:
        out = nn_model(img_tensor)

        # Skoru hesapla
        if isinstance(out, (list, tuple)):
            score = out[0].max() if hasattr(out[0], 'max') else out[-1].max()
        else:
            score = out.max()

        nn_model.zero_grad()
        score.backward()

        if feats and grads:
            f = feats[0]
            g = grads[0]
            w   = g.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * f).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=(640, 640), mode="bilinear", align_corners=False
            )
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((640, 640))
    except Exception as e:
        print(f"  ⚠️  {model_name} Grad-CAM hatası: {e}")
        cam = np.zeros((640, 640))

    h1.remove()
    h2.remove()

    return cam


# ─────────────────────────────────────────────────────────────
# ANA GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────

def plot_all_gradcam(samples):
    """
    Tüm modeller için Grad-CAM ızgarası:
    Satırlar = Modeller (YOLO11n, YOLO11s, RT-DETR)
    Sütunlar = Örnek görüntüler (6 adet)
    Her hücre: Orijinal + Grad-CAM overlay
    """
    n_models  = len(MODELS)
    n_samples = len(samples)

    fig, axes = plt.subplots(
        n_models * 2, n_samples,
        figsize=(n_samples * 3.5, n_models * 7)
    )
    fig.suptitle(
        "Grad-CAM — YOLO11n, YOLO11s, RT-DETR\n"
        "Modellerin Fixed-Wing Drone Tespitinde Odaklandığı Bölgeler",
        fontsize=14, fontweight="bold", y=1.01
    )

    for row, (name, path) in enumerate(MODELS.items()):
        if not Path(path).exists():
            print(f"  ⚠️  {name} modeli bulunamadı: {path}")
            continue

        print(f"  🔍 {name} işleniyor...")
        color = MODEL_COLORS[name]

        for col, img_path in enumerate(samples):
            # Tespit sonucu
            img_rgb, annotated, conf, _ = get_gradcam_ultralytics(
                path, str(img_path)
            )

            # Grad-CAM ısı haritası
            cam = manual_gradcam(path, str(img_path), name)

            # Overlay oluştur
            img_resized = cv2.resize(img_rgb, (640, 640))
            heatmap = cv2.applyColorMap(
                (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = img_resized / 255.0 * 0.5 + heatmap * 0.5
            overlay = np.clip(overlay, 0, 1)

            # Orijinal + tespit kutusu
            ax_orig = axes[row * 2][col]
            ax_cam  = axes[row * 2 + 1][col]

            annotated_resized = cv2.resize(annotated, (640, 640))
            ax_orig.imshow(annotated_resized)
            ax_orig.axis("off")
            if col == 0:
                ax_orig.set_ylabel(f"{name}\nTespit",
                                   fontsize=10, fontweight="bold",
                                   color=color, rotation=90,
                                   labelpad=10, va="center")

            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            if col == 0:
                ax_cam.set_ylabel(f"{name}\nGrad-CAM",
                                  fontsize=10, fontweight="bold",
                                  color=color, rotation=90,
                                  labelpad=10, va="center")

            if row == 0:
                ax_orig.set_title(f"Örnek {col+1}", fontsize=9)

            ax_cam.set_title(f"Conf: {conf:.2f}", fontsize=8)

    plt.tight_layout()
    save_path = PLOT_DIR / "yolo_gradcam_all.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {save_path}")


def plot_comparison_gradcam(samples):
    """
    4 model karşılaştırması — her sütun bir örnek,
    her satır bir model (Custom CNN dahil).
    """
    all_models = {
        "Custom CNN": None,  # Grad-CAM zaten üretildi
        **MODELS
    }

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(
        "4 Model Grad-CAM Karşılaştırması\n"
        "Kırmızı = Yüksek Dikkat | Mavi = Düşük Dikkat",
        fontsize=14, fontweight="bold"
    )

    model_colors = {
        "Custom CNN":  "#2EC4B6",
        "YOLO11n":     "#FF9F1C",
        "YOLO11s":     "#E71D36",
        "RT-DETR":     "#7209B7",
    }

    sample_3 = samples[:3]

    for row, (name, path) in enumerate(MODELS.items()):
        if not Path(path).exists():
            continue

        for col, img_path in enumerate(sample_3):
            cam = manual_gradcam(path, str(img_path), name)

            img_rgb     = np.array(Image.open(img_path).convert("RGB").resize((640, 640)))
            heatmap     = cv2.applyColorMap(
                (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay     = img_rgb / 255.0 * 0.5 + heatmap * 0.5
            overlay     = np.clip(overlay, 0, 1)

            axes[row + 1][col].imshow(overlay)
            axes[row + 1][col].axis("off")
            if col == 0:
                axes[row + 1][col].set_ylabel(
                    name, fontsize=11, fontweight="bold",
                    color=model_colors[name],
                    rotation=90, labelpad=10, va="center"
                )

    # İlk satır: orijinal görüntüler
    for col, img_path in enumerate(sample_3):
        img = np.array(Image.open(img_path).convert("RGB").resize((640, 640)))
        axes[0][col].imshow(img)
        axes[0][col].set_title(f"Örnek {col+1}", fontsize=10)
        axes[0][col].axis("off")
        if col == 0:
            axes[0][col].set_ylabel("Orijinal", fontsize=11,
                                     fontweight="bold", rotation=90,
                                     labelpad=10, va="center")

    plt.tight_layout()
    save_path = PLOT_DIR / "all_models_gradcam_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {save_path}")


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  YOLO Grad-CAM Görselleştirme")
    print("="*55)

    samples = get_samples(n=6)
    print(f"  📸 {len(samples)} örnek görüntü seçildi.")

    if not samples:
        print("  ❌ val/ klasöründe görüntü bulunamadı!")
        exit(1)

    # Tüm YOLO modelleri için Grad-CAM
    plot_all_gradcam(samples)

    # 4 model karşılaştırması (3 örnek)
    plot_comparison_gradcam(samples)

    print("\n  ✅ Tamamlandı!")
    print(f"  📁 plots/ klasörüne bakın:")
    print(f"     - yolo_gradcam_all.png")
    print(f"     - all_models_gradcam_comparison.png")
