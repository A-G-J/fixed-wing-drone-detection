"""
=========================================================
  Custom CNN — İyileştirilmiş Eğitim (Yerel CPU)
  Önceki eğitimdeki sorunlar düzeltildi:
  - Daha düşük learning rate
  - Daha iyi detection head
  - Anchor tabanlı tahmin
  - Daha uzun sabır (patience)
=========================================================
  Çalıştırma:
      python retrain_custom_cnn.py
=========================================================
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
CFG = {
    "train_img": "train",
    "train_lbl": "train",
    "val_img":   "val",
    "val_lbl":   "val",
    "img_size":  416,       # 640'dan küçük — CPU için daha hızlı
    "batch_size": 8,        # CPU için küçük batch
    "epochs":    50,
    "lr":        1e-4,      # Daha düşük LR — daha stabil
    "patience":  15,        # Daha uzun sabır
    "save_path": "runs/detect/CustomCNN_v2/best.pt",
    "plot_dir":  Path("plots"),
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# 1. MİMARİ — İyileştirilmiş Custom CNN
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class ResBlock(nn.Module):
    """Artık bağlantılı blok — gradyan akışını iyileştirir."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch // 2, k=1, p=0),
            ConvBNReLU(ch // 2, ch, k=3, p=1),
        )
    def forward(self, x):
        return x + self.block(x)

class ImprovedDetectionHead(nn.Module):
    """
    İyileştirilmiş detection head.
    Her grid hücresi için [x, y, w, h, objectness] tahmin eder.
    Birden fazla grid pozisyonunda tahmin yapar.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_ch, 256, k=3, p=1),
            ResBlock(256),
            ConvBNReLU(256, 128, k=3, p=1),
            nn.Conv2d(128, 5, 1),  # 5 = x,y,w,h,obj
        )
    def forward(self, x):
        return self.head(x)  # (B, 5, H, W)

class ImprovedCustomCNN(nn.Module):
    """
    İyileştirilmiş Custom CNN Dedektörü.
    - ResBlock'lar ile daha iyi gradyan akışı
    - Çoklu ölçek özellik birleştirme
    - Daha güçlü detection head
    """
    def __init__(self):
        super().__init__()
        # Backbone
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=2, p=1),   # 416→208
            ConvBNReLU(32, 64, k=3, s=1, p=1),
        )
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 128, k=3, s=2, p=1),  # 208→104
            ResBlock(128),
            ResBlock(128),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 256, k=3, s=2, p=1), # 104→52
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 512, k=3, s=2, p=1), # 52→26
            ResBlock(512),
            ResBlock(512),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(512, 1024, k=3, s=2, p=1), # 26→13
            ResBlock(1024),
        )
        # Detection head — 13×13 grid
        self.det_head = ImprovedDetectionHead(1024)
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
        return self.det_head(x)  # (B, 5, 13, 13)

    def get_cam_layer(self):
        return self.stage4[-1].block[-1]


# ─────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────

class DroneDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=416, augment=False):
        self.img_dir  = Path(img_dir)
        self.lbl_dir  = Path(lbl_dir)
        self.img_size = img_size
        self.augment  = augment
        self.imgs     = sorted(self.img_dir.glob("*.png"))

        base_tf = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        aug_tf = [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        self.transform = transforms.Compose(aug_tf if augment else base_tf)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img    = Image.open(img_path).convert("RGB")
        img    = self.transform(img)
        target = torch.zeros(5)

        if lbl_path.exists():
            lines = lbl_path.read_text().strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split()
                if len(parts) >= 5:
                    target = torch.tensor([
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]), 1.0
                    ])
        return img, target


# ─────────────────────────────────────────────────────────────
# 3. KAYIP FONKSİYONU
# ─────────────────────────────────────────────────────────────

def detection_loss(pred, target):
    """
    İyileştirilmiş detection kaybı:
    - Tüm grid hücrelerinde objectness loss
    - En yüksek confidence'a sahip hücrede box loss
    """
    B, C, H, W = pred.shape

    # En yüksek objectness hücresini bul
    obj_map = pred[:, 4, :, :]  # (B, H, W)
    flat    = obj_map.reshape(B, -1)
    best_idx = flat.argmax(dim=1)  # (B,)
    best_h   = best_idx // W
    best_w   = best_idx % W

    # Tahmin edilen box (en iyi hücre)
    pred_boxes = torch.stack([
        pred[i, :4, best_h[i], best_w[i]] for i in range(B)
    ])  # (B, 4)
    pred_obj = torch.stack([
        pred[i, 4, best_h[i], best_w[i]] for i in range(B)
    ])  # (B,)

    # Box loss — sadece nesne olan örneklerde
    obj_mask = target[:, 4] > 0.5
    box_loss = torch.tensor(0.0)
    if obj_mask.sum() > 0:
        box_loss = nn.SmoothL1Loss()(
            torch.sigmoid(pred_boxes[obj_mask]),
            target[obj_mask, :4]
        )

    # Objectness loss — tüm örnekler
    obj_loss = nn.BCEWithLogitsLoss()(pred_obj, target[:, 4])

    return 5.0 * box_loss + obj_loss, box_loss.item(), obj_loss.item()


# ─────────────────────────────────────────────────────────────
# 4. EĞİTİM
# ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cpu")
    print(f"\n  🖥️  Cihaz: CPU")
    print(f"  📐 Görüntü boyutu: {CFG['img_size']}×{CFG['img_size']}")

    # Veri yükleyiciler
    train_ds = DroneDataset(CFG["train_img"], CFG["train_lbl"],
                             CFG["img_size"], augment=True)
    val_ds   = DroneDataset(CFG["val_img"],   CFG["val_lbl"],
                             CFG["img_size"], augment=False)

    # CPU için az sayıda örnek kullan (hız için)
    from torch.utils.data import Subset
    train_subset = Subset(train_ds, range(min(1000, len(train_ds))))
    val_subset   = Subset(val_ds,   range(min(300,  len(val_ds))))

    train_loader = DataLoader(train_subset, batch_size=CFG["batch_size"],
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=CFG["batch_size"],
                               shuffle=False, num_workers=0)

    model     = ImprovedCustomCNN().to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  📊 Toplam parametre: {total_p:,}")

    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=CFG["epochs"], eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    patience_cnt = 0
    start = time.time()

    print(f"\n  {'Epoch':<8} {'TR Loss':<12} {'VL Loss':<12} {'LR':<12}")
    print("  " + "-"*44)

    for epoch in range(1, CFG["epochs"] + 1):
        # Eğitim
        model.train()
        tr_loss = 0.0
        for imgs, targets in train_loader:
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
                pred = model(imgs)
                loss, _, _ = detection_loss(pred, targets)
                vl_loss += loss.item()
        vl_loss /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["lr"].append(lr)

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
            tag = "✅"
        else:
            patience_cnt += 1
            tag = ""

        print(f"  {epoch:02d}/{CFG['epochs']}   "
              f"{tr_loss:10.4f}   {vl_loss:10.4f}   "
              f"{lr:.2e}  {tag}")

        if patience_cnt >= CFG["patience"]:
            print(f"\n  ⏹  Early stopping — epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\n  ⏱  Süre: {elapsed/60:.1f} dk")

    # Kaydet
    Path(CFG["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CFG["save_path"])
    print(f"  💾 Kaydedildi: {CFG['save_path']}")

    # Eğitim eğrisi
    plot_curves(history, elapsed)

    return model, best_state, history


def plot_curves(history, elapsed):
    ep  = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ep, history["train_loss"], color="#2EC4B6", lw=2, label="Eğitim")
    ax1.plot(ep, history["val_loss"],   color="#2EC4B6", lw=2,
             linestyle="--", alpha=0.7, label="Doğrulama")
    ax1.set_title("Custom CNN v2 — Kayıp Eğrisi", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ep, history["lr"], color="#2EC4B6", lw=2)
    ax2.set_title("Öğrenme Hızı (Cosine Annealing)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR (log)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Custom CNN v2 — Eğitim Süresi: {elapsed/60:.1f} dk",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    CFG["plot_dir"].mkdir(exist_ok=True)
    plt.savefig(CFG["plot_dir"] / "custom_cnn_v2_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: plots/custom_cnn_v2_curves.png")


# ─────────────────────────────────────────────────────────────
# 5. GRAD-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.feats = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out): self.feats = out
    def _bwd(self, _, __, g):   self.grads = g[0]

    def __call__(self, x):
        self.model.eval()
        out   = self.model(x)
        # En yüksek objectness skorunu hedef al
        score = out[:, 4, :, :].max()
        self.model.zero_grad()
        score.backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.feats).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(
            cam, size=(416, 416), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        conf = float(torch.sigmoid(out[:, 4, :, :].max()).item())
        return cam, conf


def generate_gradcam(model):
    """6 örnek görüntü için Grad-CAM üretir."""
    print("\n  🎨 Grad-CAM görüntüleri oluşturuluyor...")
    target_layer = model.stage4[-1].block[-1]
    gcam = GradCAM(model, target_layer)

    val_imgs = sorted(Path(CFG["val_img"]).glob("*.png"))
    samples  = [p for p in val_imgs
                if (Path(CFG["val_lbl"]) / (p.stem + ".txt")).exists()][:6]

    transform = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle(
        "Custom CNN v2 — Grad-CAM Isı Haritaları\n"
        "(Kırmızı = Yüksek Dikkat, Mavi = Düşük Dikkat)",
        fontsize=13, fontweight="bold"
    )

    for col, img_path in enumerate(samples):
        img_pil    = Image.open(img_path).convert("RGB").resize(
            (CFG["img_size"], CFG["img_size"])
        )
        img_tensor = transform(img_pil).unsqueeze(0).requires_grad_(True)

        try:
            cam, conf = gcam(img_tensor)
        except Exception as e:
            print(f"  ⚠️  {img_path.name}: {e}")
            continue

        orig_np = np.array(img_pil) / 255.0
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(orig_np * 0.55 + heatmap * 0.45, 0, 1)

        axes[0][col].imshow(orig_np)
        axes[0][col].set_title(f"Orijinal", fontsize=8)
        axes[0][col].axis("off")

        axes[1][col].imshow(overlay)
        axes[1][col].set_title(f"Grad-CAM\nConf:{conf:.2f}", fontsize=8)
        axes[1][col].axis("off")

    plt.tight_layout()
    save_path = CFG["plot_dir"] / "custom_cnn_v2_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Kaydedildi: {save_path}")


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Custom CNN v2 — İyileştirilmiş Eğitim")
    print("="*50)

    model, best_state, history = train()

    # En iyi ağırlıkları yükle
    model.load_state_dict(best_state)

    # Grad-CAM üret
    generate_gradcam(model)

    print("\n  ✅ Tamamlandı!")
    print("  📁 plots/ klasörüne bakın")
