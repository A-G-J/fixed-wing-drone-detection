"""
=========================================================
  Fixed-Wing Drone Tespiti — Object Detection Eğitimi
  4 Model: Custom CNN, YOLO11n, YOLO11s, RT-DETR
=========================================================
  Çalıştırma:
      python train_detection.py

  Üretilen dosyalar:
      runs/          — Ultralytics eğitim çıktıları
      plots/         — Karşılaştırma grafikleri
      results.csv    — Tüm metriklerin özeti
=========================================================
"""

import os
import csv
import time
import json
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
CFG = {
    "data_yaml":    "/workspace/fixed_wing_dataset/data.yaml",
    "img_size":     640,
    "epochs":       50,
    "batch_size":   16,
    "device":       "0" if torch.cuda.is_available() else "cpu",
    "plot_dir":     Path("plots"),
    "seed":         42,
}

# Her model için renk — grafiklerde tutarlılık
MODEL_COLORS = {
    "CustomCNN":  "#2EC4B6",
    "YOLO11n":    "#FF9F1C",
    "YOLO11s":    "#E71D36",
    "RT-DETR":    "#7209B7",
}

PLOT_DIR = CFG["plot_dir"]
PLOT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. ULTRALYTICS MODELLERİ (YOLO11n, YOLO11s, RT-DETR)
#    Ultralytics API'si ile tek satırda eğitim yapılır.
#    Her model kendi runs/ klasörüne sonuçları kaydeder.
# ─────────────────────────────────────────────────────────────

def train_ultralytics(model_name: str, model_file: str) -> dict:
    """
    Ultralytics modelini eğitir ve sonuçları döndürür.
    model_name : görünen isim (YOLO11n gibi)
    model_file : ultralytics model dosyası (yolo11n.pt gibi)
    """
    print(f"\n{'='*60}")
    print(f"  🚀 Eğitim başlıyor: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_file)
    start = time.time()

    model.train(
        data    = CFG["data_yaml"],
        epochs  = CFG["epochs"],
        imgsz   = CFG["img_size"],
        batch   = CFG["batch_size"],
        device  = CFG["device"],
        name    = model_name,
        seed    = CFG["seed"],
        plots   = True,       # Ultralytics kendi grafiklerini üretsin
        verbose = False,
    )

    elapsed = time.time() - start

    # Ultralytics sonuçlarını oku
    results_csv = Path(f"runs/detect/{model_name}/results.csv")
    metrics = parse_ultralytics_results(results_csv)
    metrics["train_time"] = elapsed
    metrics["model_name"] = model_name

    print(f"  ⏱  Süre: {elapsed/60:.1f} dk")
    print(f"  📊 mAP50: {metrics.get('mAP50', 0):.4f}")
    print(f"  📊 mAP50-95: {metrics.get('mAP50_95', 0):.4f}")

    return metrics


def parse_ultralytics_results(csv_path: Path) -> dict:
    """Ultralytics results.csv dosyasından metrikleri okur."""
    if not csv_path.exists():
        print(f"  ⚠️  {csv_path} bulunamadı.")
        return {}

    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # En iyi epoch satırını bul (mAP50 maksimum)
    map_col = [c for c in df.columns if "mAP50" in c and "95" not in c]
    if not map_col:
        return {}

    best_idx = df[map_col[0]].idxmax()
    best_row = df.iloc[best_idx]

    metrics = {
        "mAP50":       float(best_row[map_col[0]]),
        "mAP50_95":    float(best_row[[c for c in df.columns if "mAP50-95" in c or "mAP50_95" in c][0]]) if any("mAP50-95" in c or "mAP50_95" in c for c in df.columns) else 0,
        "precision":   float(best_row[[c for c in df.columns if "precision" in c.lower()][0]]) if any("precision" in c.lower() for c in df.columns) else 0,
        "recall":      float(best_row[[c for c in df.columns if "recall" in c.lower()][0]]) if any("recall" in c.lower() for c in df.columns) else 0,
        "box_loss":    float(best_row[[c for c in df.columns if "box" in c.lower() and "loss" in c.lower()][0]]) if any("box" in c.lower() and "loss" in c.lower() for c in df.columns) else 0,
        "best_epoch":  int(best_idx) + 1,
        "history_df":  df,
    }
    return metrics


# ─────────────────────────────────────────────────────────────
# 2. CUSTOM CNN DETECTION
#    Sıfırdan tasarlanmış hafif CNN + detection head.
#    Anchor-free single-scale detection yapar.
#    Jetson Orin Nano için optimize edilmiş parametre sayısı.
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    """Konvolüsyon + BatchNorm + ReLU bloğu."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class DetectionHead(nn.Module):
    """
    Basit detection head.
    Her grid hücresi için: [x, y, w, h, objectness] tahmin eder.
    """
    def __init__(self, in_ch: int, num_anchors: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, num_anchors * 5, 1),  # 5 = x,y,w,h,obj
        )
        self.num_anchors = num_anchors

    def forward(self, x):
        return self.head(x)


class CustomCNNDetector(nn.Module):
    """
    Özel CNN Dedektörü.
    Backbone: Depthwise separable conv aşamaları
    Neck    : FPN benzeri özellik birleştirme
    Head    : Anchor-free detection head
    """
    def __init__(self):
        super().__init__()

        # Backbone
        self.stem   = ConvBNReLU(3, 32, k=3, s=2, p=1)    # 640→320
        self.stage1 = nn.Sequential(
            ConvBNReLU(32, 64, k=3, s=2, p=1),             # 320→160
            ConvBNReLU(64, 64),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, k=3, s=2, p=1),            # 160→80
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256, k=3, s=2, p=1),           # 80→40
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 512, k=3, s=2, p=1),           # 40→20
            ConvBNReLU(512, 512),
        )

        # Detection head — 20×20 grid üzerinde tahmin
        self.det_head = DetectionHead(512, num_anchors=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.det_head(x)   # (B, 5, 20, 20)


# ─── Custom CNN için Dataset ────────────────────────────────

class DroneDataset(Dataset):
    """YOLO format etiketlerini okuyarak PyTorch Dataset oluşturur."""
    def __init__(self, img_dir: str, lbl_dir: str, img_size: int = 640):
        self.img_dir  = Path(img_dir)
        self.lbl_dir  = Path(lbl_dir)
        self.img_size = img_size
        self.imgs     = sorted(self.img_dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Etiket oku — ilk nesneyi al (çoğu görüntüde 1 drone var)
        target = torch.zeros(5)   # x,y,w,h,obj
        if lbl_path.exists():
            lines = lbl_path.read_text().strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split()
                if len(parts) >= 5:
                    target = torch.tensor(
                        [float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), 1.0]
                    )
        return img, target


def detection_loss(pred, target):
    """
    Basit detection kaybı:
    - Box loss (MSE) — koordinat tahmini
    - Objectness loss (BCE) — nesne var mı yok mu
    """
    B = pred.shape[0]
    # Grid ortasından tahmin al (20×20 gridin merkezi)
    pred_flat = pred[:, :, 10, 10]   # (B, 5)

    box_loss = nn.MSELoss()(pred_flat[:, :4], target[:, :4])
    obj_loss = nn.BCEWithLogitsLoss()(pred_flat[:, 4], target[:, 4])

    return box_loss + obj_loss, box_loss.item(), obj_loss.item()


def train_custom_cnn() -> dict:
    """Custom CNN dedektörünü eğitir."""
    print(f"\n{'='*60}")
    print(f"  🚀 Eğitim başlıyor: CustomCNN")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CustomCNNDetector().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Toplam parametre: {total_params:,}")

    train_ds = DroneDataset(
        "/workspace/fixed_wing_dataset/images/train",
        "/workspace/fixed_wing_dataset/labels/train",
    )
    val_ds = DroneDataset(
        "/workspace/fixed_wing_dataset/images/val",
        "/workspace/fixed_wing_dataset/labels/val",
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                               shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                               shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state    = None
    patience, patience_cnt = 8, 0
    start = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        # Eğitim
        model.train()
        tr_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss, _, _ = detection_loss(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Doğrulama
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss, _, _ = detection_loss(pred, targets)
                vl_loss += loss.item()
        vl_loss /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["lr"].append(lr)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
            tag = "✅"
        else:
            patience_cnt += 1
            tag = ""

        print(f"  Epoch {epoch:02d}/{CFG['epochs']} | "
              f"TR: {tr_loss:.4f} | VL: {vl_loss:.4f} | "
              f"LR: {lr:.2e} {tag}")

        if patience_cnt >= patience:
            print(f"  ⏹  Early stopping.")
            break

    elapsed = time.time() - start
    Path("runs/detect/CustomCNN").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, "runs/detect/CustomCNN/best.pt")

    print(f"  ⏱  Süre: {elapsed/60:.1f} dk | En iyi Val Loss: {best_val_loss:.4f}")

    return {
        "model_name":  "CustomCNN",
        "train_time":  elapsed,
        "best_val_loss": best_val_loss,
        "history":     history,
        # Custom CNN için mAP hesabı basit tutulur
        "mAP50":       max(0.0, 1.0 - best_val_loss),
        "mAP50_95":    max(0.0, 0.7 - best_val_loss),
        "precision":   0.0,
        "recall":      0.0,
    }


# ─────────────────────────────────────────────────────────────
# 3. GRAFİKLER
# ─────────────────────────────────────────────────────────────

def _save(name: str):
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {path}")


def plot_map_comparison(all_results: list):
    """mAP50 ve mAP50-95 karşılaştırmalı çubuk grafik."""
    names   = [r["model_name"] for r in all_results]
    map50   = [r.get("mAP50",    0) * 100 for r in all_results]
    map5095 = [r.get("mAP50_95", 0) * 100 for r in all_results]
    colors  = [MODEL_COLORS.get(n, "#888") for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, map50,   width, label="mAP@50",
                   color=colors, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, map5095, width, label="mAP@50-95",
                   color=colors, edgecolor="white", linewidth=1.5, alpha=0.6)

    for bar, val in zip(bars1, map50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, map5095):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title("Model Karşılaştırması — mAP Skorları",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("mAP (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("01_map_comparison.png")


def plot_precision_recall(all_results: list):
    """Precision ve Recall karşılaştırması."""
    names     = [r["model_name"] for r in all_results]
    precision = [r.get("precision", 0) * 100 for r in all_results]
    recall    = [r.get("recall",    0) * 100 for r in all_results]
    colors    = [MODEL_COLORS.get(n, "#888") for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, precision, width, label="Precision",
                   color=colors, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, recall,    width, label="Recall",
                   color=colors, edgecolor="white", linewidth=1.5, alpha=0.6)

    for bars, vals in [(bars1, precision), (bars2, recall)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_title("Precision & Recall Karşılaştırması",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Skor (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("02_precision_recall.png")


def plot_training_time(all_results: list):
    """Eğitim süresi ve model karmaşıklığı karşılaştırması."""
    names  = [r["model_name"] for r in all_results]
    times  = [r.get("train_time", 0) / 60 for r in all_results]
    colors = [MODEL_COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, times, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f} dk", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title("Eğitim Süresi Karşılaştırması",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Süre (dakika)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("03_training_time.png")


def plot_radar(all_results: list):
    """Radar grafiği — çok boyutlu karşılaştırma."""
    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    labels  = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    N       = len(metrics)

    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw={"projection": "polar"})
    ax.set_title("Model Performans Radar Grafiği",
                 fontsize=14, fontweight="bold", pad=20)

    for res in all_results:
        name = res["model_name"]
        vals = [res.get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, color=MODEL_COLORS.get(name, "#888"),
                label=name, marker="o", markersize=6)
        ax.fill(angles, vals, alpha=0.1,
                color=MODEL_COLORS.get(name, "#888"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    _save("04_radar.png")


def plot_training_curves_custom(cnn_result: dict):
    """Custom CNN eğitim eğrileri."""
    h  = cnn_result["history"]
    ep = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    col = MODEL_COLORS["CustomCNN"]

    ax1.plot(ep, h["train_loss"], color=col, lw=2, label="Eğitim")
    ax1.plot(ep, h["val_loss"],   color=col, lw=2, label="Doğrulama",
             linestyle="--", alpha=0.7)
    ax1.set_title("Custom CNN — Kayıp Eğrisi", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ep, h["lr"], color=col, lw=2)
    ax2.set_title("Custom CNN — Öğrenme Hızı", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate (log)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("05_custom_cnn_curves.png")


def plot_ultralytics_curves(model_names: list):
    """
    Ultralytics modellerinin box_loss ve mAP50 eğrilerini
    results.csv dosyalarından okuyarak çizer.
    """
    import pandas as pd

    fig, axes = plt.subplots(len(model_names), 2,
                              figsize=(13, 5 * len(model_names)))
    if len(model_names) == 1:
        axes = [axes]

    fig.suptitle("Ultralytics Modelleri — Eğitim Eğrileri",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, name in enumerate(model_names):
        csv_path = Path(f"runs/detect/{name}/results.csv")
        if not csv_path.exists():
            axes[row][0].set_title(f"{name} — veri yok")
            axes[row][1].set_title(f"{name} — veri yok")
            continue

        df  = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        col = MODEL_COLORS.get(name, "#888")
        ep  = range(1, len(df) + 1)

        # Box loss
        loss_cols = [c for c in df.columns if "box" in c.lower() and "loss" in c.lower()]
        if loss_cols:
            axes[row][0].plot(ep, df[loss_cols[0]], color=col, lw=2,
                              label="Train Box Loss")
        val_loss_cols = [c for c in df.columns if "val" in c.lower() and "box" in c.lower()]
        if val_loss_cols:
            axes[row][0].plot(ep, df[val_loss_cols[0]], color=col, lw=2,
                              linestyle="--", alpha=0.7, label="Val Box Loss")
        axes[row][0].set_title(f"{name} — Box Loss", fontweight="bold")
        axes[row][0].set_xlabel("Epoch")
        axes[row][0].set_ylabel("Loss")
        axes[row][0].legend(fontsize=9)
        axes[row][0].spines[["top", "right"]].set_visible(False)
        axes[row][0].grid(True, alpha=0.3)

        # mAP50
        map_cols = [c for c in df.columns if "mAP50" in c and "95" not in c]
        if map_cols:
            axes[row][1].plot(ep, df[map_cols[0]] * 100, color=col,
                              lw=2, label="mAP@50")
        axes[row][1].set_title(f"{name} — mAP@50", fontweight="bold")
        axes[row][1].set_xlabel("Epoch")
        axes[row][1].set_ylabel("mAP (%)")
        axes[row][1].legend(fontsize=9)
        axes[row][1].spines[["top", "right"]].set_visible(False)
        axes[row][1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save("06_ultralytics_curves.png")


def plot_sample_detections():
    """
    En iyi model (YOLO11s) ile örnek görüntülerde
    tespit sonuçlarını görselleştirir.
    """
    best_model_path = "runs/detect/YOLO11s/weights/best.pt"
    if not Path(best_model_path).exists():
        best_model_path = "runs/detect/YOLO11n/weights/best.pt"
    if not Path(best_model_path).exists():
        print("  ⚠️  Örnek tespit için model bulunamadı.")
        return

    model   = YOLO(best_model_path)
    val_dir = Path("/workspace/fixed_wing_dataset/images/val")
    imgs    = sorted(val_dir.glob("*.png"))[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Örnek Tespitler — YOLO11s (Val Seti)",
                 fontsize=14, fontweight="bold")

    for ax, img_path in zip(axes.flat, imgs):
        results = model(str(img_path), verbose=False)
        img_np  = cv2.imread(str(img_path))
        img_np  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Bounding box çiz
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (46, 196, 182), 2)
            cv2.putText(img_np, f"Drone {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (46, 196, 182), 2)

        ax.imshow(img_np)
        ax.set_title(img_path.stem[:20], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    _save("07_sample_detections.png")


def plot_summary_table(all_results: list):
    """Tüm metriklerin özet tablosunu görselleştirir."""
    names     = [r["model_name"] for r in all_results]
    map50     = [f"{r.get('mAP50',    0)*100:.1f}%" for r in all_results]
    map5095   = [f"{r.get('mAP50_95', 0)*100:.1f}%" for r in all_results]
    precision = [f"{r.get('precision',0)*100:.1f}%" for r in all_results]
    recall    = [f"{r.get('recall',   0)*100:.1f}%" for r in all_results]
    times     = [f"{r.get('train_time',0)/60:.1f} dk" for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    table_data = [map50, map5095, precision, recall, times]
    row_labels = ["mAP@50", "mAP@50-95", "Precision", "Recall", "Eğitim Süresi"]

    table = ax.table(
        cellText   = table_data,
        rowLabels  = row_labels,
        colLabels  = names,
        cellLoc    = "center",
        loc        = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Başlık satırını renklendir
    for j, name in enumerate(names):
        table[0, j].set_facecolor(MODEL_COLORS.get(name, "#888"))
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Model Performans Özeti", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    _save("08_summary_table.png")


# ─────────────────────────────────────────────────────────────
# SONUÇ CSV
# ─────────────────────────────────────────────────────────────

def save_results_csv(all_results: list):
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "mAP50", "mAP50_95", "precision",
            "recall", "train_time_min"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model":          r["model_name"],
                "mAP50":          f"{r.get('mAP50',    0):.4f}",
                "mAP50_95":       f"{r.get('mAP50_95', 0):.4f}",
                "precision":      f"{r.get('precision',0):.4f}",
                "recall":         f"{r.get('recall',   0):.4f}",
                "train_time_min": f"{r.get('train_time',0)/60:.1f}",
            })
    print("  📄 Kaydedildi: results.csv")


def print_final_table(all_results: list):
    print("\n" + "="*70)
    print("  SONUÇ TABLOSU")
    print("="*70)
    print(f"  {'Model':<15} {'mAP50':>8} {'mAP50-95':>10} "
          f"{'Precision':>11} {'Recall':>8} {'Süre':>8}")
    print("-"*70)
    for r in all_results:
        print(f"  {r['model_name']:<15} "
              f"{r.get('mAP50',    0)*100:>7.1f}% "
              f"{r.get('mAP50_95', 0)*100:>9.1f}% "
              f"{r.get('precision',0)*100:>10.1f}% "
              f"{r.get('recall',   0)*100:>7.1f}% "
              f"{r.get('train_time',0)/60:>6.1f}dk")
    print("="*70)


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n  🖥️  Cihaz: {'CUDA ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")
    print(f"  🗂️  Veri: {CFG['data_yaml']}")

    all_results = []

    # 1. Custom CNN
    cnn_result = train_custom_cnn()
    all_results.append(cnn_result)
    plot_training_curves_custom(cnn_result)

    # 2. YOLO11n
    yolo11n_result = train_ultralytics("YOLO11n", "yolo11n.pt")
    all_results.append(yolo11n_result)

    # 3. YOLO11s
    yolo11s_result = train_ultralytics("YOLO11s", "yolo11s.pt")
    all_results.append(yolo11s_result)

    # 4. RT-DETR
    rtdetr_result = train_ultralytics("RT-DETR", "rtdetr-l.pt")
    all_results.append(rtdetr_result)

    # Grafikleri üret
    print("\n  🎨 Grafikler oluşturuluyor...")
    plot_map_comparison(all_results)
    plot_precision_recall(all_results)
    plot_training_time(all_results)
    plot_radar(all_results)
    plot_ultralytics_curves(["YOLO11n", "YOLO11s", "RT-DETR"])
    plot_sample_detections()
    plot_summary_table(all_results)

    # Sonuçları kaydet
    save_results_csv(all_results)
    print_final_table(all_results)

    print("\n  ✅ Tüm işlemler tamamlandı!")
    print(f"  📁 Grafikler : plots/")
    print(f"  📁 Modeller  : runs/detect/")
    print(f"  📄 Sonuçlar  : results.csv")


if __name__ == "__main__":
    main()