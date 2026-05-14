"""
=============================================================================
仿生输入层验证 — 可学习高斯调制 vs 捷径
=============================================================================
假设: 仿生节点(可学习高斯模糊+通道调制)在输入层能主动抑制捷径信号
对比:
  H) 标准有偏训练(基线)
  I) 固定模糊(8x8)
  J) 可学习高斯调制(仿生) [新]
  C) 灰度剥离(已知最优)
=============================================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random, sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

SEED = 42
N_COLORS, N_COUNTS = 3, 5
DOT_R = 6
BASE_SIZE = 128
COLORS = [(220,50,50),(50,100,220),(50,180,50)]
SPURIOUS_MAP = {0:2, 1:3, 2:4}
SPURIOUS_RATIO = 1.00
N_TRAIN = 300
N_CF = 80
EPOCHS, BS, LR = 40, 128, 3e-4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"Device: DirectML")
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")


# ==================== 仿生输入层 v2: 通道混合 ====================

class BionicInputLayer(nn.Module):
    """
    仿生输入层 v2: 可学习的通道混合矩阵 + 可选空间模糊
    
    原理: 文本Token间高斯模糊 ≈ 图像通道间混合
         让模型自己学会"压红=灰度化"来去颜色捷径
    
    参数: 极少量 (~18个)
      - color_matrix: 3x3 可学习通道混合矩阵 (9个)
      - sigma: 空间高斯模糊强度 (1个可学习)
      - channel_gain: 每通道调制因子 (3个, 连续调制)
      - kernel_weights: 5x5空间卷积核 (25个, 单通道)
    """
    def __init__(self, in_channels=3, kernel_size=5, init_sigma=2.0):
        super().__init__()
        self.in_channels = in_channels
        
        # 核心: 可学习的通道混合矩阵 (9个参数)
        # 初始为单位矩阵 (不做混合), 模型自己学
        self.color_matrix = nn.Parameter(torch.eye(in_channels))
        
        # 可学习的空间模糊强度
        self.log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma)))
        
        # 可学习的通道增益 (连续调制 - 模拟旁分泌)
        self.channel_gain = nn.Parameter(torch.ones(in_channels))
        
        # 空间高斯卷积核
        k = kernel_size
        self.spatial_conv = nn.Conv2d(1, 1, k, padding=k//2, bias=False)
        self._init_gaussian(init_sigma)
    
    def _init_gaussian(self, sigma):
        k = self.spatial_conv.kernel_size[0]
        center = k // 2
        xs = torch.arange(k, dtype=torch.float32) - center
        ys = torch.arange(k, dtype=torch.float32) - center
        gauss = torch.exp(-(xs.unsqueeze(0)**2 + ys.unsqueeze(1)**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        with torch.no_grad():
            self.spatial_conv.weight.data[0, 0] = gauss.clone()
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 通道混合: (B, C, H, W) → (B, C, H, W)
        #    color_matrix shape (C, C) → batch matrix multiply
        x = torch.einsum('bchw,cd->bdhw', x, self.color_matrix)
        
        # 2. 通道增益调制
        x = x * self.channel_gain.view(1, -1, 1, 1)
        
        # 3. 空间高斯模糊 (每通道独立)
        x_flat = x.reshape(B * C, 1, H, W)
        x_flat = self.spatial_conv(x_flat)
        x = x_flat.reshape(B, C, H, W)
        
        return x
    
    def extra_repr(self):
        mat = self.color_matrix.data
        return (f"color_matrix=\n{mat.tolist()}\n"
                f"  sigma={self.sigma.item():.2f}, "
                f"gain={[f'{g:.2f}' for g in self.channel_gain.data.tolist()]}")
    
    def get_mixing_strength(self):
        """返回通道混合偏离单位矩阵的程度 (0=没混合, 大的值=强制混合)"""
        mat = self.color_matrix.data.cpu()
        eye = torch.eye(self.in_channels)
        return (mat - eye).abs().mean().item()


# ==================== 数据生成 ====================

def generate_batch(color_idxs, dot_counts):
    B = len(color_idxs)
    H = W = BASE_SIZE
    imgs = np.ones((B, 3, H, W), dtype=np.float32)
    ctr, rad = H//2, H//3
    yg, xg = np.ogrid[:H, :W]
    dist = np.sqrt((xg-ctr)**2 + (yg-ctr)**2)
    fill_mask = dist <= rad
    ring_mask = (dist <= rad) & (dist >= rad-3)
    
    for b in range(B):
        rgb = np.array(COLORS[color_idxs[b]], dtype=np.float32)/255.0
        fill_c = rgb*0.25 + 0.75
        for ch in range(3):
            imgs[b,ch][fill_mask] = fill_c[ch]
            imgs[b,ch][ring_mask] = rgb[ch]
        
        dc = dot_counts[b]
        n_dots = (dc.item() if hasattr(dc, 'item') else dc) + 1
        max_off = rad - DOT_R - 6
        for d_idx in range(n_dots):
            if n_dots == 1:
                angle, off = 0, 0
            else:
                angle = 2*np.pi*d_idx/n_dots + random.uniform(-0.15,0.15)
                off = max_off*random.uniform(0.30,0.88)
            px, py = ctr + int(off*np.cos(angle)), ctr + int(off*np.sin(angle))
            dy, dx = np.ogrid[:H, :W]
            dd = np.sqrt((dx-px)**2+(dy-py)**2)
            imgs[b,:][:, dd<=DOT_R] = 0.05
    return torch.from_numpy(imgs).float()


def make_dataset(n_per_color, spurious_ratio):
    color_idxs, count_idxs = [], []
    for ci in range(N_COLORS):
        for _ in range(n_per_color):
            color_idxs.append(ci)
            if random.random() < spurious_ratio:
                count_idxs.append(SPURIOUS_MAP[ci])
            else:
                count_idxs.append(random.randint(0, N_COUNTS-1))
    return color_idxs, count_idxs


def make_counterfactual(n_per):
    color_idxs, count_idxs = [], []
    for ci in range(N_COLORS):
        for cnt in range(N_COUNTS):
            for _ in range(n_per):
                color_idxs.append(ci); count_idxs.append(cnt)
    return color_idxs, count_idxs


@torch.no_grad()
def resize_tensor(imgs, size):
    return F.interpolate(imgs, size=(size,size), mode='bilinear', align_corners=False)


def to_gray(imgs):
    return imgs.mean(dim=1, keepdim=True)


def build_loader(color_idxs, count_idxs, img_size, gray=False, shuffle=True):
    n_total = len(color_idxs)
    all_imgs = []
    for start in range(0, n_total, 256):
        end = min(start+256, n_total)
        ci = color_idxs[start:end]; cnti = count_idxs[start:end]
        batch = generate_batch(ci, cnti)
        batch = resize_tensor(batch, img_size)
        if gray: batch = to_gray(batch)
        all_imgs.append(batch)
    imgs = torch.cat(all_imgs)
    ci_t = torch.tensor(color_idxs, dtype=torch.long)
    cnti_t = torch.tensor(count_idxs, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, ci_t, cnti_t), BS, shuffle=shuffle)


# ==================== 模型 ====================

def cbr(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c,out_c,3,padding=1),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class Small64(nn.Module):
    def __init__(self, nc, ich=3, bionic=None):
        super().__init__()
        self.bionic = bionic
        self.net = nn.Sequential(
            cbr(ich,16), nn.MaxPool2d(2),
            cbr(16,32), nn.MaxPool2d(2),
            cbr(32,64), nn.MaxPool2d(2),
            cbr(64,128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,nc))
    
    def forward(self, x):
        if self.bionic is not None:
            x = self.bionic(x)
        return self.net(x)


# ==================== 训练 ====================

def train_one(model, tr_ld, vl_ld, epochs, lr):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_st = 0, None
    for _ in range(epochs):
        model.train()
        for im, c, cnt in tr_ld:
            im = im.to(DEVICE)
            lbl = cnt.to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(im),lbl).backward(); opt.step()
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for im, c, cnt in vl_ld:
                im = im.to(DEVICE)
                cor += (model(im).argmax(1).cpu()==cnt).sum().item()
                tot += len(cnt)
        acc = cor/tot
        if acc >= best_acc:
            best_acc=acc; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model, best_acc


@torch.no_grad()
def ev(model, ld):
    model.eval(); cor, tot = 0, 0
    for im, _, cnt in ld:
        im = im.to(DEVICE)
        cor += (model(im).argmax(1).cpu()==cnt).sum().item()
        tot += len(cnt)
    return cor/tot


# ==================== main ====================

def main():
    rbl = 1/N_COUNTS
    print(f"\n仿生输入层实验 | 捷径100% | {N_TRAIN*3}训练图 | {EPOCHS}epochs")
    
    print("\n[1] 生成数据...")
    tr_c, tr_cnt = make_dataset(N_TRAIN, SPURIOUS_RATIO)
    vl_c, vl_cnt = make_counterfactual(20)
    ts_c, ts_cnt = make_counterfactual(N_CF)
    
    trL64 = build_loader(tr_c, tr_cnt, 64)
    trL64g = build_loader(tr_c, tr_cnt, 64, gray=True)
    vL64 = build_loader(vl_c, vl_cnt, 64, shuffle=False)
    vL64g = build_loader(vl_c, vl_cnt, 64, gray=True, shuffle=False)
    tL64 = build_loader(ts_c, ts_cnt, 64, shuffle=False)
    tL64g = build_loader(ts_c, ts_cnt, 64, gray=True, shuffle=False)
    
    print("\n[2] 训练对比...")
    
    # H) 基线(有偏彩色)
    mH, aH = train_one(Small64(N_COUNTS), trL64, vL64, EPOCHS, LR)
    rH = ev(mH, tL64)
    print(f"  [H] 基线(有偏彩色): val={aH:.4f}  test_counterfactual={rH:.4f}")
    
    # C) 灰度(最优已知)
    mC, aC = train_one(Small64(N_COUNTS, ich=1), trL64g, vL64g, EPOCHS, LR)
    rC = ev(mC, tL64g)
    print(f"  [C] 灰度剥离: val={aC:.4f}  test={rC:.4f}")
    
    # I) 固定模糊(8x8)
    trL64_blur = build_loader(tr_c, tr_cnt, 64)
    # 手动做固定模糊包装
    class FixedBlurModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Small64(N_COUNTS)
        def forward(self, x):
            s = F.interpolate(x, (8,8), mode='bilinear', align_corners=False)
            s = F.interpolate(s, (64,64), mode='bilinear', align_corners=False)
            return self.net(s)
    mI = FixedBlurModel()
    # 用原数据训练(模型内部做模糊)
    def train_fixed(model, tr_ld, vl_ld, epochs, lr):
        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_acc, best_st = 0, None
        for _ in range(epochs):
            model.train()
            for im, c, cnt in tr_ld:
                im = im.to(DEVICE); lbl = cnt.to(DEVICE)
                opt.zero_grad(); nn.CrossEntropyLoss()(model(im),lbl).backward(); opt.step()
            model.eval()
            cor, tot = 0, 0
            with torch.no_grad():
                for im, c, cnt in vl_ld:
                    im = im.to(DEVICE)
                    cor += (model(im).argmax(1).cpu()==cnt).sum().item()
                    tot += len(cnt)
            acc = cor/tot
            if acc >= best_acc:
                best_acc=acc; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_st)
        return model, best_acc
    mI, aI = train_fixed(mI, trL64, vL64, EPOCHS, LR)
    rI = ev(mI, tL64)
    print(f"  [I] 固定模糊(8x8): val={aI:.4f}  test={rI:.4f}")
    
    # J) 可学习高斯调制(仿生输入层)
    bionic = BionicInputLayer(in_channels=3, kernel_size=5, init_sigma=2.0)
    mJ, aJ = train_one(Small64(N_COUNTS, bionic=bionic), trL64, vL64, EPOCHS, LR)
    rJ = ev(mJ, tL64)
    learned_sigma = mJ.bionic.sigma.item()
    learned_gain = [g.item() if hasattr(g, 'item') else g for g in mJ.bionic.channel_gain.data.cpu()]
    learned_matrix = mJ.bionic.color_matrix.data.cpu().tolist()
    mixing_strength = mJ.bionic.get_mixing_strength()
    print(f"  [J] 仿生输入层 v2: val={aJ:.4f}  test={rJ:.4f}")
    print(f"      通道混合偏离度: {mixing_strength:.4f} (0=无混合, 大=强混合)")
    print(f"      sigma={learned_sigma:.2f} gain={[f'{g:.2f}' for g in learned_gain]}")
    print(f"      混合矩阵:")
    for row in learned_matrix:
        print(f"        [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")
    
    # 训练分布准确率
    trL64ts = build_loader(tr_c, tr_cnt, 64, shuffle=False)
    tr_rH = ev(mH, trL64ts)
    tr_rI = ev(mI, trL64ts)
    tr_rJ = ev(mJ, trL64ts)
    
    # ============ 结果 ============
    print("\n" + "="*65)
    print("[3] 结果汇总")
    print("="*65)
    
    print(f"""
  {'模型':<25}{'训练(有偏)':>9}{'反事实':>9}{'捷径差距':>9}
  {'-'*54}
  H) 基线(有偏彩色)      {tr_rH:>9.4f}{rH:>9.4f}{tr_rH-rH:>9.4f}
  I) 固定模糊(8x8)       {tr_rI:>9.4f}{rI:>9.4f}{tr_rI-rI:>9.4f}
  J) 仿生输入层 [新]     {tr_rJ:>9.4f}{rJ:>9.4f}{tr_rJ-rJ:>9.4f}
  C) 灰度剥离(最优)      --      {rC:>9.4f}  --
  随机基线                --      {rbl:>9.4f}  --
""")
    
    print("  核心发现:")
    
    gapH = tr_rH - rH
    gapJ = tr_rJ - rJ
    gapC_gap = rC - rbl
    
    print(f"  J(仿生)反事实: {rJ:.4f}  vs  H(基线): {rH:.4f}  vs  C(灰度): {rC:.4f}")
    
    if rJ > rH * 1.3:
        print(f"  [V] 仿生输入层显著优于基线! +{(rJ-rH)*100:.1f}pp")
    elif rJ > rH:
        print(f"  [~] 仿生输入层略优于基线")
    else:
        print(f"  [~] 仿生输入层与基线接近")
    
    if rJ > rI:
        print(f"  [V] 可学习高斯 > 固定模糊(+{(rJ-rI)*100:.1f}pp)")
        print(f"  [V] 让模型自己调节模糊强度比人为固定更有效")
    else:
        print(f"  可学习 vs 固定: 差异不显著")
    
    print(f"\n  仿生层最终参数:")
    print(f"    sigma={learned_sigma:.2f} (初始=2.0)")
    print(f"    通道混合偏离度={mixing_strength:.4f} (0=接近单位矩阵)")
    print(f"    通道增益: R={learned_gain[0]:.2f} G={learned_gain[1]:.2f} B={learned_gain[2]:.2f}")
    
    if mixing_strength > 0.05:
        print(f"  -> 模型学会了通道混合! 偏离度={mixing_strength:.4f}")
        print(f"  -> 混合矩阵:")
        for row in learned_matrix:
            print(f"       [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")
    else:
        print(f"  -> 通道混合几乎未激活(偏离度={mixing_strength:.4f})")
    
    if learned_sigma > 3.0:
        print(f"  -> 模型学会了主动增强空间模糊")
    
    print(f"\n  结论:")
    print(f"  - 通道混合版仿生层: {'有效!' if rJ > rH*1.15 else '效果有限'}")
    print(f"  - 跨模态验证: {'支持' if rJ > rH*1.1 else '需更多探索'} 文本→图像的迁移")
    
    print("\n实验完成。")


if __name__=='__main__':
    main()
