"""
=========================================================
  Custom CNN — Değerlendirme & Grad-CAM Görselleştirme
  val/ klasöründeki görüntüler üzerinde çalışır.
=========================================================
  Çalıştırma:
      python evaluate_custom_cnn.py
=========================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2

# ─────────────────────────────────────────────────────────────
# Model tanımı — train_detection.py ile aynı
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_anchors=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, num_anchors * 5, 1),
        )
    def forward(self, x):
        return self.head(x)

class CustomCNNDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = ConvBNReLU(3, 32, k=3, s=2, p=1)
        self.stage1 = nn.Sequential(
            ConvBNReLU(32, 64, k=3, s=2, p=1),
            ConvBNReLU(64, 64),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, k=3, s=2, p=1),
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256, k=3, s=2, p=1),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 512, k=3, s=2, p=1),
            ConvBNReLU(512, 512),
        )
        self.det_head = DetectionHead(512, num_anchors=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.det_head(x)


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Custom CNN için Grad-CAM ısı haritası üretir.
    Modelin görüntünün hangi bölgesine odaklandığını gösterir.
    """
    def __init__(self, model, target_layer):
        self.model  = model
        self.grads  = None
        self.feats  = None
        target_layer.register_forward_hook(self._save_feats)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, _, __, output):
        self.feats = output

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        out   = self.model(x)
        score = out[:, 4, 10, 10]  # Objectness skoru
        self.model.zero_grad()
        score.backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.feats).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = nn.functional.interpolate(
            cam, size=(640, 640), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(torch.sigmoid(score).item())


# ─────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def unnorm(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)

def load_label(lbl_path):
    """YOLO format etiketini okur — ilk nesnenin bbox'ını döndürür."""
    if not Path(lbl_path).exists():
        return None
    lines = Path(lbl_path).read_text().strip().split("\n")
    if not lines or not lines[0]:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    return [float(p) for p in parts[1:5]]  # x,y,w,h (normalized)

def yolo_to_xyxy(box, W=640, H=640):
    """YOLO format bbox'ı piksel koordinatlarına çevirir."""
    cx, cy, w, h = box
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)
    return x1, y1, x2, y2

def compute_iou(box1, box2):
    """İki bbox arasındaki IoU değerini hesaplar."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


# ─────────────────────────────────────────────────────────────
# Ana değerlendirme
# ─────────────────────────────────────────────────────────────

def main():
    device    = torch.device("cpu")
    model_path = "runs/detect/CustomCNN/best.pt"
    val_img   = Path("val")
    val_lbl   = Path("val")
    out_dir   = Path("plots")
    out_dir.mkdir(exist_ok=True)

    print("  📦 Model yükleniyor...")
    model = CustomCNNDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  ✅ Model yüklendi!")

    # Grad-CAM kurulumu — son konvolüsyon katmanı
    gcam = GradCAM(model, model.stage4[-1][0])

    # Val görüntülerini listele
    imgs = sorted(val_img.glob("*.png"))[:200]  # İlk 200 görüntü yeterli
    print(f"  📊 {len(imgs)} görüntü değerlendiriliyor...")

    # ── Metrik hesaplama ──
    tp, fp, fn = 0, 0, 0
    iou_scores = []
    conf_threshold = 0.5

    for img_path in imgs:
        lbl_path = val_lbl / (img_path.stem + ".txt")
        gt_box   = load_label(lbl_path)

        img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor)

        conf = torch.sigmoid(out[:, 4, 10, 10]).item()
        pred_box_raw = out[0, :4, 10, 10].detach().cpu().numpy()
        pred_box = [
            float(torch.sigmoid(torch.tensor(pred_box_raw[0])).item()),
            float(torch.sigmoid(torch.tensor(pred_box_raw[1])).item()),
            float(torch.sigmoid(torch.tensor(pred_box_raw[2])).item()),
            float(torch.sigmoid(torch.tensor(pred_box_raw[3])).item()),
        ]

        pred_detected = conf >= conf_threshold
        gt_exists     = gt_box is not None

        if pred_detected and gt_exists:
            pred_xyxy = yolo_to_xyxy(pred_box)
            gt_xyxy   = yolo_to_xyxy(gt_box)
            iou       = compute_iou(pred_xyxy, gt_xyxy)
            iou_scores.append(iou)
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
        elif pred_detected and not gt_exists:
            fp += 1
        elif not pred_detected and gt_exists:
            fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou  = np.mean(iou_scores) if iou_scores else 0

    print(f"\n  📊 Custom CNN Sonuçları:")
    print(f"     Precision : {precision:.4f} ({precision*100:.1f}%)")
    print(f"     Recall    : {recall:.4f} ({recall*100:.1f}%)")
    print(f"     F1 Score  : {f1:.4f} ({f1*100:.1f}%)")
    print(f"     Mean IoU  : {mean_iou:.4f} ({mean_iou*100:.1f}%)")
    print(f"     TP: {tp} | FP: {fp} | FN: {fn}")

    # ── Grad-CAM görselleştirme ──
    print("\n  🎨 Grad-CAM görüntüleri oluşturuluyor...")
    sample_imgs = [p for p in imgs if load_label(val_lbl / (p.stem + ".txt"))][:6]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle("Custom CNN — Grad-CAM Isı Haritaları\n(Modelin Odaklandığı Bölgeler)",
                 fontsize=14, fontweight="bold")

    for col, img_path in enumerate(sample_imgs):
        img_pil    = Image.open(img_path).convert("RGB").resize((640, 640))
        img_tensor = transform(img_pil).unsqueeze(0).requires_grad_(True)

        try:
            cam, conf = gcam(img_tensor)
        except Exception as e:
            print(f"  ⚠️  {img_path.name}: {e}")
            continue

        orig_np = np.array(img_pil) / 255.0

        # Isı haritası
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = orig_np * 0.55 + heatmap * 0.45
        overlay = np.clip(overlay, 0, 1)

        # GT bbox çiz
        lbl = load_label(val_lbl / (img_path.stem + ".txt"))
        if lbl:
            x1, y1, x2, y2 = yolo_to_xyxy(lbl)
            orig_with_box = orig_np.copy()
            cv2.rectangle(
                (orig_with_box * 255).astype(np.uint8),
                (x1, y1), (x2, y2), (0, 255, 0), 2
            )

        axes[0][col].imshow(orig_np)
        axes[0][col].set_title(f"Orijinal\n{img_path.stem[:12]}", fontsize=7)
        axes[0][col].axis("off")

        axes[1][col].imshow(overlay)
        axes[1][col].set_title(f"Grad-CAM\nConf: {conf:.2f}", fontsize=7)
        axes[1][col].axis("off")

    plt.tight_layout()
    save_path = out_dir / "custom_cnn_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {save_path}")

    # ── Sonuçları kaydet ──
    with open("custom_cnn_results.txt", "w") as f:
        f.write(f"Custom CNN Evaluation Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")
        f.write(f"Mean IoU  : {mean_iou:.4f}\n")
        f.write(f"TP: {tp} | FP: {fp} | FN: {fn}\n")
    print("  💾 Kaydedildi: custom_cnn_results.txt")
    print("\n  ✅ Tamamlandı!")


if __name__ == "__main__":
    main()
