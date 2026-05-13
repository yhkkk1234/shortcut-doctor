"""
=============================================================================
渐进模糊防捷径 — CIFAR级验证 (合成10类图案+色块捷径)
=============================================================================
验证: 渐进模糊能否逼迫CNN学习真实图像特征而非捷径色块?
捷径: 每类图案配一个独特色块(角标), 100%关联
测试: 干净测试集(无色块)上谁更准
=============================================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random, os, sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_CLASSES = 10
IMG_SIZE = 32
N_TRAIN = 100         # per class, total 1000 (极少数据逼模型走捷径)
N_VAL = 50             # per class
N_TEST = 100           # per class
EPOCHS = 30
BS = 64
LR = 1e-3

# 每类的唯一捷径色块
SHORTCUT_COLORS = [(255,0,0),(255,128,0),(255,255,0),(0,255,0),(0,255,255),
                   (0,0,255),(128,0,255),(255,0,255),(128,128,128),(200,100,50)]

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"Device: DirectML (AMD) - {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")


# ==================== 合成数据 ====================

def generate_pattern(class_id, size=IMG_SIZE):
    """生成 32x32 相似但不同的几何图案"""
    img = np.ones((size, size, 3), dtype=np.float32) * 0.85
    h = size
    
    # 所有图案共用"中心有东西"的模板, 区别在于中心图案类型和方向
    # 前5类: 不同方向的条纹
    r = h // 3
    center = h // 2
    
    if class_id < 5:
        # 条纹族: 不同角度和间距
        angles = [0, 30, 60, 90, 120]
        spacing = [6, 5, 7, 4, 6]
        a = np.radians(angles[class_id])
        sp = spacing[class_id]
        yg, xg = np.ogrid[:h, :h]
        rotated = (xg - center) * np.cos(a) + (yg - center) * np.sin(a)
        for i in range(-h//sp, h//sp):
            band = np.abs(rotated - i * sp) < sp * 0.4
            img[band] = 0.25
    elif class_id < 9:
        # 圆/椭圆族: 不同长宽比和位置
        ratios = [(1,1), (0.7,1), (0.5,1), (0.7,0.7)]
        offsets = [0, -2, 2, -1]
        rx, ry = ratios[class_id - 5]
        off = offsets[class_id - 5]
        yg, xg = np.ogrid[:h, :h]
        ellipse = ((xg - center) ** 2) / (r * rx) ** 2 + ((yg - center - off) ** 2) / (r * ry) ** 2
        img[ellipse <= 1] = 0.25
    else:
        # 方块族: 不同位置和大小的方块
        sz = r + (class_id - 9) * 2
        off_x = (class_id - 9) * 2 - 1
        x1, y1 = center - sz + off_x, center - sz
        x2, y2 = center + sz + off_x, center + sz
        img[y1:y2, x1:x2, :] = 0.25
    
    # 加噪增强真实感
    img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img.transpose(2, 0, 1)


def add_shortcut(img, class_id):
    """在右下角加 8x8 色块作为捷径 (占图6%, 极度显眼)"""
    c = np.array(SHORTCUT_COLORS[class_id], dtype=np.float32) / 255.0
    img[:, IMG_SIZE-9:IMG_SIZE-1, IMG_SIZE-9:IMG_SIZE-1] = c.reshape(3, 1, 1)
    return img


def make_dataset(n_per_class, with_shortcut=True):
    imgs, labels = [], []
    for cls in range(N_CLASSES):
        for _ in range(n_per_class):
            img = generate_pattern(cls)
            if with_shortcut:
                img = add_shortcut(img.copy(), cls)
            imgs.append(img)
            labels.append(cls)
    imgs_t = torch.from_numpy(np.stack(imgs)).float()
    labels_t = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(imgs_t, labels_t)


def blur_tensor(imgs, strength):
    size_map = {'strong': 4, 'medium': 8, 'mild': 16}
    target = size_map.get(strength, strength)
    orig = imgs.shape[-1]
    small = F.interpolate(imgs, size=(target, target), mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(orig, orig), mode='bilinear', align_corners=False)


class BlurLoader:
    def __init__(self, dataset, schedule):
        self.dataset = dataset
        self.schedule = schedule  # [(epoch_end, strength), ...]
    
    def get_loader(self, epoch):
        for end_ep, strength in self.schedule:
            if epoch < end_ep:
                if strength is None:
                    return DataLoader(self.dataset, BS, shuffle=True)
                return BlurredDataLoader(self.dataset, BS, strength)
        return DataLoader(self.dataset, BS, shuffle=True)


class BlurredDataLoader:
    def __init__(self, dataset, batch_size, blur_strength):
        self.base_loader = DataLoader(dataset, batch_size, shuffle=True)
        self.blur_strength = blur_strength
    
    def __iter__(self):
        for imgs, labels in self.base_loader:
            yield blur_tensor(imgs, self.blur_strength), labels
    
    def __len__(self):
        return len(self.base_loader)


# ==================== 模型 ====================

class CifarCNN(nn.Module):
    """小型CNN, 适配32x32输入, ~2M参数"""
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        return self.net(x)


# ==================== 训练 ====================

def train_basic(model, train_loader, val_loader, epochs, lr):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_st = 0, None
    
    for ep in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(imgs), labels).backward()
            opt.step()
        
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                cor += (model(imgs).argmax(1).cpu() == labels).sum().item()
                tot += len(labels)
        acc = cor / tot
        if acc >= best_acc:
            best_acc = acc
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_st)
    return model, best_acc


def train_progressive(model, blur_loader, clean_val_loader, epochs, lr):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_st = 0, None
    
    for ep in range(epochs):
        tr_ld = blur_loader.get_loader(ep)
        model.train()
        for imgs, labels in tr_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(imgs), labels).backward()
            opt.step()
        
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for imgs, labels in clean_val_loader:
                imgs = imgs.to(DEVICE)
                cor += (model(imgs).argmax(1).cpu() == labels).sum().item()
                tot += len(labels)
        acc = cor / tot
        if acc >= best_acc:
            best_acc = acc
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_st)
    return model, best_acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    cor, tot = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        cor += (model(imgs).argmax(1).cpu() == labels).sum().item()
        tot += len(labels)
    return cor / tot


# ==================== Main ====================

def main():
    print(f"\n  数据: {N_CLASSES}类相似几何图案 @ {IMG_SIZE}x{IMG_SIZE}, {N_TRAIN*N_CLASSES}训练 / {N_TEST*N_CLASSES}测试")
    print(f"  捷径: 每类配独特色块(右下4x4), 100%关联")
    
    print("\n[1] 生成数据...")
    train_shortcut = make_dataset(N_TRAIN, with_shortcut=True)
    train_clean = make_dataset(N_TRAIN, with_shortcut=False)
    val_clean = make_dataset(N_VAL, with_shortcut=False)
    test_clean = make_dataset(N_TEST, with_shortcut=False)
    test_shortcut = make_dataset(N_TEST, with_shortcut=True)
    
    print(f"  训练(有捷径): {len(train_shortcut)}  训练(干净): {len(train_clean)}")
    print(f"  验证(干净): {len(val_clean)}  测试(干净): {len(test_clean)}")
    
    trL_shortcut = DataLoader(train_shortcut, BS, shuffle=True)
    trL_clean = DataLoader(train_clean, BS, shuffle=True)
    vlL = DataLoader(val_clean, BS)
    tsL_clean = DataLoader(test_clean, BS)
    tsL_shortcut = DataLoader(test_shortcut, BS)
    
    # 固定模糊
    trL_fixed = BlurredDataLoader(train_shortcut, BS, 'medium')
    
    # 渐进模糊: 前15 epoch strong, 15-30 medium, 30-40 mild
    blur_loader = BlurLoader(train_shortcut, [(10, 'strong'), (18, 'medium'), (25, None)])
    
    print("\n[2] 训练模型...")
    
    print("\n  [A] 基线: 有捷径训练 → 干净测试")
    mA, _ = train_basic(CifarCNN(N_CLASSES), trL_shortcut, vlL, EPOCHS, LR)
    rA_s = evaluate(mA, tsL_shortcut)
    rA_c = evaluate(mA, tsL_clean)
    print(f"      捷径测试: {rA_s:.4f}  干净测试: {rA_c:.4f}  捷径差距: {rA_s-rA_c:.4f}")
    
    print("\n  [B] 上界: 无捷径训练 → 干净测试")
    mB, _ = train_basic(CifarCNN(N_CLASSES), trL_clean, vlL, EPOCHS, LR)
    rB_c = evaluate(mB, tsL_clean)
    print(f"      干净测试: {rB_c:.4f}  (上界)")
    
    print("\n  [C] 固定模糊(8x8): 有捷径训练 → 干净测试")
    mC, _ = train_basic(CifarCNN(N_CLASSES), trL_fixed, vlL, EPOCHS, LR)
    rC_s = evaluate(mC, tsL_shortcut)
    rC_c = evaluate(mC, tsL_clean)
    print(f"      捷径测试: {rC_s:.4f}  干净测试: {rC_c:.4f}  捷径差距: {rC_s-rC_c:.4f}")
    
    print("\n  [D] 渐进模糊(4→8→16): 有捷径训练 → 干净测试")
    mD, _ = train_progressive(CifarCNN(N_CLASSES), blur_loader, vlL, EPOCHS, LR)
    rD_s = evaluate(mD, tsL_shortcut)
    rD_c = evaluate(mD, tsL_clean)
    print(f"      捷径测试: {rD_s:.4f}  干净测试: {rD_c:.4f}  捷径差距: {rD_s-rD_c:.4f}")
    
    # ============ 结论 ============
    print("\n" + "="*65)
    print("[3] 结果汇总")
    print("="*65)
    
    print(f"""
  {'模型':<30} {'捷径测试':>10} {'干净测试':>10} {'捷径差距':>10}
  {'-'*62}
  {'A) 有捷径(纯色块依赖性)':<30} {rA_s:>10.4f} {rA_c:>10.4f} {rA_s-rA_c:>10.4f}
  {'B) 无捷径(上界)':<30} {'--':>10} {rB_c:>10.4f} {'--':>10}
  {'C) 固定模糊(8x8)':<30} {rC_s:>10.4f} {rC_c:>10.4f} {rC_s-rC_c:>10.4f}
  {'D) 渐进模糊(4→8→16)':<30} {rD_s:>10.4f} {rD_c:>10.4f} {rD_s-rD_c:>10.4f}
""")
    
    print("  关键发现:")
    
    # 捷径依赖程度: A在干净集上的表现
    print(f"  A(纯捷径)在干净集上: {rA_c:.4f} (随机基线={1/N_CLASSES:.2f})")
    if rA_c < 0.2:
        print(f"  → 模型几乎完全依赖色块, 干了集完全崩溃")
    elif rA_c < rB_c * 0.5:
        print(f"  → 模型重度依赖捷径")
    
    # 固定模糊
    print(f"\n  C(固定模糊)在干净集上: {rC_c:.4f} vs A: {rA_c:.4f}")
    if rC_c > rA_c * 1.3:
        print(f"  [V] 固定模糊提升了干净集准确率 +{(rC_c-rA_c)*100:.1f}%")
    else:
        print(f"  [~] 固定模糊改善有限")
    
    # 渐进模糊
    print(f"\n  D(渐进模糊)在干净集上: {rD_c:.4f} vs A: {rA_c:.4f}")
    if rD_c > rA_c * 1.5:
        print(f"  [V] 渐进模糊大幅提升干净集准确率 +{(rD_c-rA_c)*100:.1f}%")
    elif rD_c > rA_c * 1.2:
        print(f"  [V] 渐进模糊有效提升干净集准确率")
    
    if rD_c > rC_c * 1.05:
        print(f"  [V] 渐进模糊({rD_c:.4f}) > 固定模糊({rC_c:.4f})")
        print(f"  [V] 模拟发育过程优于固定模糊, 跨实验一致!")
    
    # 对上界
    print(f"\n  上界B(无捷径): {rB_c:.4f}")
    print(f"  D(渐进模糊)达到上界的 {rD_c/rB_c*100:.1f}%" if rB_c > 0 else "")
    
    print("\n实验完成。")


if __name__ == '__main__':
    main()
