"""
=============================================================================
渐进模糊防捷径 — 真实 CIFAR-100 实验
=============================================================================
验证: 注入颜色边框捷径后, 渐进模糊能否迫使ResNet学习图像内容?
捷径: 每类配唯一色边框(4px), 100%关联
测试: 干净CIFAR-100测试集(无边框)
=============================================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random, os, sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_CLASSES = 100
BORDER_WIDTH = 4
EPOCHS = 30
BS = 256
LR = 1e-3
NUM_WORKERS = 4

# 为每类生成唯一色边框
np.random.seed(99)
SHORTCUT_COLORS = np.random.randint(50, 230, (N_CLASSES, 3))  # 避开太暗/太亮的色

# 设备: CUDA优先, 其次DirectML, 最后CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
        print(f"Device: DirectML (AMD) - {torch_directml.device_name(0)}")
    except:
        DEVICE = torch.device('cpu')
        print(f"Device: CPU")

# ==================== 捷径注入 ====================

class ShortcutDataset(Dataset):
    """包装CIFAR数据集, 给每张图加颜色边框捷径"""
    def __init__(self, base_dataset):
        self.base = base_dataset
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        # img: (3,32,32) tensor, normalize前
        c = torch.tensor(SHORTCUT_COLORS[label], dtype=torch.float32) / 255.0
        c = c.view(3, 1, 1).expand(3, BORDER_WIDTH, img.shape[2])
        img[:, :BORDER_WIDTH, :] = c
        img[:, -BORDER_WIDTH:, :] = c
        c_v = c.transpose(1, 2)
        img[:, :, :BORDER_WIDTH] = c_v
        img[:, :, -BORDER_WIDTH:] = c_v
        return img, label


class BlurWrapper(Dataset):
    """训练时自动模糊的数据集包装器"""
    def __init__(self, base_dataset, blur_strength):
        self.base = base_dataset
        self.strength = blur_strength
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.strength:
            size_map = {'strong': 8, 'medium': 16, 'mild': 24}
            t = size_map.get(self.strength, self.strength)
            img = img.unsqueeze(0)
            img = F.interpolate(img, (t, t), mode='bilinear', align_corners=False)
            img = F.interpolate(img, (32, 32), mode='bilinear', align_corners=False)
            img = img.squeeze(0)
        return img, label


class ProgressiveBlurWrapper(Dataset):
    """渐进模糊: 按epoch动态切换模糊度"""
    def __init__(self, base_dataset, schedule):
        self.base = base_dataset
        self.schedule = schedule  # [(end_epoch, strength), ...]
        self._current_strength = 'strong'
    
    def set_epoch(self, epoch):
        for end_ep, strength in self.schedule:
            if epoch < end_ep:
                self._current_strength = strength
                return
        self._current_strength = None
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        s = self._current_strength
        if s:
            size_map = {'strong': 8, 'medium': 16, 'mild': 24}
            t = size_map.get(s, s)
            img = img.unsqueeze(0)
            img = F.interpolate(img, (t, t), mode='bilinear', align_corners=False)
            img = F.interpolate(img, (32, 32), mode='bilinear', align_corners=False)
            img = img.squeeze(0)
        return img, label


# ==================== 模型 ====================

class SmallResNet(nn.Module):
    """ResNet-18 适配CIFAR (32x32)"""
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.inplanes = 64
        
        # Stem: 3x3 conv for 32x32 (no 7x7)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, planes, blocks, stride):
        layers = []
        # 下采样block
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ==================== 训练 ====================

def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss, cor, tot = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(imgs), labels)
        loss.backward(); opt.step()
        total_loss += loss.item()
        cor += (model(imgs).argmax(1) == labels).sum().item()
        tot += len(labels)
    return total_loss / len(loader), cor / tot

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    cor, tot = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cor += (model(imgs).argmax(1) == labels).sum().item()
        tot += len(labels)
    return cor / tot


def run_experiment(name, train_ds, val_loader, test_clean_loader, test_shortcut_loader,
                   epochs, lr, progressive=False):
    print(f"\n  [{name}] 训练中...")
    
    model = SmallResNet(N_CLASSES).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    best_val, best_st = 0, None
    
    for ep in range(epochs):
        if progressive:
            train_ds.set_epoch(ep)
        loader = DataLoader(train_ds, BS, shuffle=True, num_workers=NUM_WORKERS)
        
        train_loss, train_acc = train_epoch(model, loader, opt, crit)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        
        if val_acc > best_val:
            best_val = val_acc
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_st)
    acc_shortcut = evaluate(model, test_shortcut_loader)
    acc_clean = evaluate(model, test_clean_loader)
    gap = acc_shortcut - acc_clean
    
    print(f"      捷径测试: {acc_shortcut:.4f}  干净测试: {acc_clean:.4f}  捷径差距: {gap:.4f}")
    return acc_shortcut, acc_clean, gap


# ==================== Main ====================

def main():
    import time
    start_t = time.time()
    
    # 输出同时写文件, 防止SSH断开丢结果
    log_file = open('result_cifar100.txt', 'w', encoding='utf-8')
    import builtins
    orig_print = builtins.print
    def tee_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        orig_print(*args, **kwargs, file=log_file)
        log_file.flush()
    builtins.print = tee_print
    
    print(f"\nCIFAR-100 渐进模糊实验  |  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"捷径: 每类4px颜色边框, 100%关联  |  测试: 干净图(无边框)")
    
    # 加载原始CIFAR-100 (ToTensor后即可)
    base_train = datasets.CIFAR100(root='./data', train=True, download=True,
                                    transform=transforms.ToTensor())
    base_test = datasets.CIFAR100(root='./data', train=False, download=True,
                                   transform=transforms.ToTensor())
    
    # 捷径版本
    train_shortcut = ShortcutDataset(base_train)
    test_shortcut = ShortcutDataset(base_test)
    
    # 干净版本(原始)
    train_clean = base_train
    test_clean = base_test
    
    # 数据加载器
    def make_loader(ds, shuf=True):
        return DataLoader(ds, BS, shuffle=shuf, num_workers=NUM_WORKERS)
    
    # 10%训练集做验证(小验证集加快评估)
    val_indices = random.sample(range(len(train_clean)), 5000)
    val_subset = torch.utils.data.Subset(train_clean, val_indices)
    
    vlL = make_loader(val_subset, shuf=False)
    tsL_clean = make_loader(test_clean, shuf=False)
    tsL_shortcut = make_loader(test_shortcut, shuf=False)
    
    print(f"  训练: {len(train_clean)}  验证: {len(val_subset)}  测试: {len(test_clean)}")
    
    # [A] 基线: 有捷径训练 → 干净测试
    rA_s, rA_c, rA_g = run_experiment("A) 有捷径训练(基线)", train_shortcut, 
                                        vlL, tsL_clean, tsL_shortcut, EPOCHS, LR)
    
    # [B] 上界: 无捷径训练
    rB_s, rB_c, rB_g = run_experiment("B) 无捷径训练(上界)", train_clean,
                                        vlL, tsL_clean, tsL_shortcut, EPOCHS, LR)
    
    # [C] 固定模糊
    train_fixed = BlurWrapper(train_shortcut, 'mild')
    rC_s, rC_c, rC_g = run_experiment("C) 固定模糊(24x24)", train_fixed,
                                        vlL, tsL_clean, tsL_shortcut, EPOCHS, LR)
    
    # [D] 渐进模糊: 前15 strong, 15-30 medium, 30-40 mild
    train_prog = ProgressiveBlurWrapper(train_shortcut, [(12, 'strong'), (22, 'medium'), (30, 'mild')])
    rD_s, rD_c, rD_g = run_experiment("D) 渐进模糊(8→16→24)", train_prog,
                                        vlL, tsL_clean, tsL_shortcut, EPOCHS, LR, progressive=True)
    
    # ============ 结论 ============
    print("\n" + "="*65)
    print("[结果汇总]")
    print("="*65)
    
    print(f"""
  {'模型':<35} {'捷径测试':>9} {'干净测试':>9} {'捷径差距':>9}
  {'-'*64}
  A) 有捷径训练(基线)        {rA_s:>9.4f} {rA_c:>9.4f} {rA_g:>9.4f}
  B) 无捷径训练(上界)        {rB_s:>9.4f} {rB_c:>9.4f} {rB_g:>9.4f}
  C) 固定模糊(24x24)         {rC_s:>9.4f} {rC_c:>9.4f} {rC_g:>9.4f}
  D) 渐进模糊(8→16→24)      {rD_s:>9.4f} {rD_c:>9.4f} {rD_g:>9.4f}
""")
    
    print("  核心发现:")
    
    gapA = abs(rA_s - rA_c)
    print(f"  A 捷径依赖: 捷径={rA_s:.4f}, 干净={rA_c:.4f}, 差距={gapA:.4f}")
    if gapA < 0.02:
        print(f"  → 捷径和干净准确率几乎一致, 模型未明显依赖捷径")
    else:
        print(f"  → 捷径贡献了 {gapA:.4f} 的准确率差异")
    
    if rD_c > rA_c * 1.01:
        improved = rD_c - rA_c
        print(f"\n  [V] D(渐进模糊)干净测试: {rD_c:.4f} > A(基线): {rA_c:.4f}")
        print(f"  [V] 渐进模糊在真实CIFAR-100场景中提升了干净集性能 +{improved*100:.1f}%")
    else:
        print(f"\n  [~] D(渐进模糊)干净测试: {rD_c:.4f} vs A(基线): {rA_c:.4f}")
        print(f"  [~] 在此配置下渐进模糊与基线接近")
    
    if rD_c > rC_c * 1.005:
        print(f"\n  [V] 渐进模糊({rD_c:.4f}) > 固定模糊({rC_c:.4f})")
    
    print(f"\n  上界B(无捷径训练): {rB_c:.4f}")
    print(f"  渐进模糊D达到上界的: {rD_c/rB_c*100:.1f}%")
    
    print("\n实验完成。")
    elapsed = (time.time() - start_t) / 60
    print(f"总耗时: {elapsed:.1f} 分钟")
    log_file.close()

if __name__ == '__main__':
    main()
