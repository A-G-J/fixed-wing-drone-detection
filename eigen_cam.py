"""
=========================================================
  YOLO11n, YOLO11s, RT-DETR — EigenCAM Görselleştirme
  pytorch-grad-cam kütüphanesi kullanılır.
  EigenCAM, detection modelleri için Grad-CAM'den daha
  güvenilir sonuçlar verir.
=========================================================
  Çalıştırma:
      python eigen_cam.py
=========================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────

MODELS = {
    "YOLO11n": "runs/detect/YOLO11n/weights/best.pt",
    "YOLO11s": "runs/detect/YOLO11s/weights/best.pt",
    "RT-DETR":  "runs/detect/RT-DETR/weights/best.pt",
}

MODEL_COLORS = {
    "YOLO11n": "#FF9F1C",
    "YOLO11s": "#E71D36",
    "RT-DETR":  "#7209B7",
}

VAL_DIR  = Path("val")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)
IMG_SIZE = 640


# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
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


def prepare_image(img_path):
    """Görüntüyü model için hazırlar."""
    img_pil = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np  = np.array(img_pil) / 255.0  # float [0,1]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_pil).unsqueeze(0)
    return img_np, tensor


def get_target_layer(model_nn, model_name):
    """Her model için uygun hedef katmanı döndürür."""
    if "RT-DETR" in model_name or "rtdetr" in model_name.lower():
        # RT-DETR için encoder çıktısı
        layers = list(model_nn.modules())
        conv_layers = [m for m in layers if isinstance(m, torch.nn.Conv2d)]
        return conv_layers[-3] if len(conv_layers) >= 3 else conv_layers[-1]
    else:
        # YOLO11 için C3k2 bloklarından sonuncusu
        target = None
        for name, module in model_nn.named_modules():
            if "C3k2" in type(module).__name__ or "C2f" in type(module).__name__:
                target = module
        if target is None:
            # Yedek: son Conv2d
            for m in model_nn.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target = m
        return target


class YOLOOutputWrapper(torch.nn.Module):
    """
    EigenCAM için YOLO çıktısını uygun formata dönüştürür.
    Detection çıktısından objectness skorlarını alır.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Çıktıyı 2D tensor olarak döndür
        if isinstance(out, (list, tuple)):
            # YOLO çıktısı: list of tensors
            result = out[0] if isinstance(out[0], torch.Tensor) else out[-1]
            if result.dim() == 3:
                # (batch, anchors, values) -> (batch, values)
                return result[:, :, 4:5].mean(dim=1)
            return result.mean(dim=-1) if result.dim() > 2 else result
        return out


# ─────────────────────────────────────────────────────────────
# EigenCAM üretimi
# ─────────────────────────────────────────────────────────────

def generate_eigencam(model_path, model_name, img_path):
    """
    Tek bir görüntü için EigenCAM ısı haritası üretir.
    """
    yolo   = YOLO(model_path)
    nn_model = yolo.model
    nn_model.eval()

    target_layer = get_target_layer(nn_model, model_name)
    if target_layer is None:
        print(f"  ⚠️  {model_name}: Hedef katman bulunamadı")
        return None, None

    img_np, tensor = prepare_image(img_path)

    try:
        cam = EigenCAM(
            model=nn_model,
            target_layers=[target_layer],
        )
        grayscale_cam = cam(input_tensor=tensor)[0]
        visualization = show_cam_on_image(
            img_np.astype(np.float32),
            grayscale_cam,
            use_rgb=True
        )
        return visualization, grayscale_cam

    except Exception as e:
        print(f"  ⚠️  {model_name} EigenCAM hatası: {e}")
        # Yedek: basit feature visualization
        try:
            with torch.no_grad():
                feats = []
                def hook(_, __, out): feats.append(out)
                h = target_layer.register_forward_hook(hook)
                nn_model(tensor)
                h.remove()

            if feats:
                f   = feats[0].squeeze(0).mean(0)
                f   = torch.relu(f).detach().cpu().numpy()
                f   = (f - f.min()) / (f.max() - f.min() + 1e-8)
                f   = cv2.resize(f, (IMG_SIZE, IMG_SIZE))
                viz = show_cam_on_image(
                    img_np.astype(np.float32), f, use_rgb=True
                )
                return viz, f
        except Exception as e2:
            print(f"  ⚠️  {model_name} yedek yöntem hatası: {e2}")
        return None, None


# ─────────────────────────────────────────────────────────────
# YOLO TESPİT SONUÇLARI
# ─────────────────────────────────────────────────────────────

def get_detection_overlay(model_path, img_path):
    """YOLO tespitini görüntü üzerine çizer."""
    yolo    = YOLO(model_path)
    results = yolo(str(img_path), verbose=False, conf=0.3)
    img_bgr = cv2.imread(str(img_path))
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    conf_val = 0.0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Koordinatları 640x640'a ölçekle
        orig_h, orig_w = results[0].orig_shape
        x1 = int(x1 * IMG_SIZE / orig_w)
        y1 = int(y1 * IMG_SIZE / orig_h)
        x2 = int(x2 * IMG_SIZE / orig_w)
        y2 = int(y2 * IMG_SIZE / orig_h)
        conf = float(box.conf[0])
        conf_val = max(conf_val, conf)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (46, 196, 182), 3)
        cv2.putText(img_rgb, f"{conf:.2f}",
                    (x1, max(y1-8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (46, 196, 182), 2)

    return img_rgb, conf_val


# ─────────────────────────────────────────────────────────────
# ANA GRAFİK — 4 Model × 3 Örnek
# ─────────────────────────────────────────────────────────────

def plot_eigencam_grid(samples):
    """
    Her model için EigenCAM ısı haritası + tespit kutusu.
    Satırlar = Modeller | Sütunlar = Örnekler
    Her satır 2 alt satır: Tespit + EigenCAM
    """
    n_models  = len(MODELS)
    n_samples = min(3, len(samples))
    s3        = samples[:n_samples]

    fig, axes = plt.subplots(
        n_models * 2, n_samples + 1,
        figsize=((n_samples + 1) * 4, n_models * 8)
    )
    fig.suptitle(
        "EigenCAM — Modellerin Odaklandığı Bölgeler\n"
        "Üst Satır: Tespit Kutusu | Alt Satır: EigenCAM Isı Haritası",
        fontsize=14, fontweight="bold", y=1.01
    )

    for row, (name, path) in enumerate(MODELS.items()):
        color = MODEL_COLORS[name]
        print(f"  🔍 {name} işleniyor...")

        if not Path(path).exists():
            print(f"  ⚠️  {name} bulunamadı")
            continue

        # Model etiketi
        axes[row * 2][0].text(
            0.5, 0.5, name,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=color,
            transform=axes[row * 2][0].transAxes
        )
        axes[row * 2][0].axis("off")
        axes[row * 2 + 1][0].axis("off")

        for col, img_path in enumerate(s3):
            print(f"    Örnek {col+1}...")

            # Tespit kutusu
            det_img, conf = get_detection_overlay(path, img_path)
            axes[row * 2][col + 1].imshow(det_img)
            axes[row * 2][col + 1].set_title(
                f"Conf: {conf:.2f}", fontsize=9
            )
            axes[row * 2][col + 1].axis("off")
            if col == 0:
                axes[row * 2][col + 1].set_ylabel(
                    "Tespit", fontsize=9, color=color, fontweight="bold"
                )

            # EigenCAM
            viz, _ = generate_eigencam(path, name, img_path)
            if viz is not None:
                axes[row * 2 + 1][col + 1].imshow(viz)
            else:
                axes[row * 2 + 1][col + 1].text(
                    0.5, 0.5, "Görselleştirme\nmevcut değil",
                    ha="center", va="center", fontsize=9
                )
            axes[row * 2 + 1][col + 1].axis("off")
            if col == 0:
                axes[row * 2 + 1][col + 1].set_ylabel(
                    "EigenCAM", fontsize=9, color=color, fontweight="bold"
                )

    # Üst başlıklar
    for col, img_path in enumerate(s3):
        axes[0][col + 1].set_title(
            f"Örnek {col+1}\n{img_path.stem[:15]}",
            fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    save_path = PLOT_DIR / "eigencam_all_models.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Kaydedildi: {save_path}")


def plot_4model_comparison(samples):
    """
    Custom CNN Grad-CAM + 3 YOLO EigenCAM karşılaştırması.
    """
    n_samples = min(3, len(samples))
    s3 = samples[:n_samples]

    fig, axes = plt.subplots(5, n_samples, figsize=(n_samples * 4.5, 22))
    fig.suptitle(
        "4 Model XAI Karşılaştırması\n"
        "Custom CNN: Grad-CAM | YOLO11n, YOLO11s, RT-DETR: EigenCAM",
        fontsize=13, fontweight="bold"
    )

    row_labels = ["Orijinal", "Custom CNN\n(Grad-CAM)",
                  "YOLO11n\n(EigenCAM)", "YOLO11s\n(EigenCAM)",
                  "RT-DETR\n(EigenCAM)"]
    row_colors = ["#333333", "#2EC4B6", "#FF9F1C", "#E71D36", "#7209B7"]

    # Satır 0: Orijinal görüntüler
    for col, img_path in enumerate(s3):
        img = np.array(Image.open(img_path).convert("RGB").resize((640, 640)))
        axes[0][col].imshow(img)
        axes[0][col].set_title(f"Örnek {col+1}", fontsize=10, fontweight="bold")
        axes[0][col].axis("off")

    # Satır 1: Custom CNN Grad-CAM (zaten üretildi, tekrar üret)
    from retrain_custom_cnn import ImprovedCustomCNN, GradCAM as CNNGCAM
    cnn_model = ImprovedCustomCNN()
    cnn_path  = "runs/detect/CustomCNN_v2/best.pt"
    if Path(cnn_path).exists():
        cnn_model.load_state_dict(
            torch.load(cnn_path, map_location="cpu")
        )
        cnn_model.eval()
        gcam = CNNGCAM(cnn_model, cnn_model.stage4[-1].block[-1])

        tf = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        for col, img_path in enumerate(s3):
            img_pil = Image.open(img_path).convert("RGB")
            tensor  = tf(img_pil).unsqueeze(0).requires_grad_(True)
            try:
                cam, conf = gcam(tensor)
                img_np  = np.array(img_pil.resize((416, 416))) / 255.0
                hm      = cv2.applyColorMap(
                    (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                hm      = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
                overlay = np.clip(img_np * 0.5 + hm * 0.5, 0, 1)
                axes[1][col].imshow(overlay)
                axes[1][col].set_title(f"Conf: {conf:.2f}", fontsize=8)
            except:
                axes[1][col].text(0.5, 0.5, "N/A",
                                   ha="center", va="center")
            axes[1][col].axis("off")

    # Satırlar 2-4: YOLO EigenCAM
    for row_idx, (name, path) in enumerate(MODELS.items()):
        row = row_idx + 2
        for col, img_path in enumerate(s3):
            viz, _ = generate_eigencam(path, name, img_path)
            if viz is not None:
                axes[row][col].imshow(viz)
            else:
                axes[row][col].text(0.5, 0.5, "N/A",
                                     ha="center", va="center")
            axes[row][col].axis("off")

    # Satır etiketleri
    for row, (label, color) in enumerate(zip(row_labels, row_colors)):
        axes[row][0].set_ylabel(
            label, fontsize=10, fontweight="bold",
            color=color, rotation=90, labelpad=15, va="center"
        )

    plt.tight_layout()
    save_path = PLOT_DIR / "4model_xai_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {save_path}")


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  EigenCAM — YOLO XAI Görselleştirme")
    print("="*55)

    samples = get_samples(n=6)
    print(f"  📸 {len(samples)} örnek seçildi.\n")

    # YOLO modelleri için EigenCAM grid
    plot_eigencam_grid(samples)

    # 4 model karşılaştırması
    plot_4model_comparison(samples)

    print("\n  ✅ Tamamlandı!")
    print("  📁 Üretilen dosyalar:")
    print("     - plots/eigencam_all_models.png")
    print("     - plots/4model_xai_comparison.png")
