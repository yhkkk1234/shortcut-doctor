"""
=============================================================================
信息带宽控制防捷径 — 实验 v4: 全局模糊化 + 渐进式训练
=============================================================================
新增验证: "模拟婴儿感官发育 — 模糊信息是否更容易学到正确骨架?"
捷径: 红→3点 蓝→4点 绿→5点 (100%关联)
=============================================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random, os, sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

SEED = 42
N_COLORS, N_COUNTS = 3, 5
DOT_R = 6
BASE_SIZE = 128
COLORS = [(220,50,50),(50,100,220),(50,180,50)]
CNAMES = ['red','blue','green']
SPURIOUS_MAP = {0:2, 1:3, 2:4}  # 红→3点,蓝→4点,绿→5点
SPURIOUS_RATIO = 1.00
N_TRAIN = 300     # per color
N_CF = 80
EPOCHS, BS, LR = 40, 128, 3e-4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"Device: DirectML (AMD) - {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

# ==================== 模糊函数 ====================

def blur_tensor(imgs, strength):
    """
    通过降分辨率再升回来模拟高斯模糊效果。
    strength: 降到的目标尺寸 (越小越模糊)
      - 'strong': 4x4  → 几乎只能看到色块
      - 'medium': 8x8  → 能看到大致结构
      - 'mild':   16x16→ 能看清形状, 细节模糊
      - 'light':  32x32→ 接近原始
    """
    size_map = {'strong': 4, 'medium': 8, 'mild': 16, 'light': 32}
    target = size_map.get(strength, strength)
    orig = imgs.shape[-1]
    small = F.interpolate(imgs, size=(target, target), mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(orig, orig), mode='bilinear', align_corners=False)

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

def build_loader(color_idxs, count_idxs, img_size, gray=False, shuffle=True, blur_strength=None):
    n_total = len(color_idxs)
    all_imgs = []
    for start in range(0, n_total, 256):
        end = min(start+256, n_total)
        ci = color_idxs[start:end]; cnti = count_idxs[start:end]
        batch = generate_batch(ci, cnti)
        batch = resize_tensor(batch, img_size)
        if gray:
            batch = to_gray(batch)
        if blur_strength is not None:
            batch = blur_tensor(batch, blur_strength)
        all_imgs.append(batch)
    
    imgs = torch.cat(all_imgs)
    ci_t = torch.tensor(color_idxs, dtype=torch.long)
    cnti_t = torch.tensor(count_idxs, dtype=torch.long)
    ds = TensorDataset(imgs, ci_t, cnti_t)
    return DataLoader(ds, BS, shuffle=shuffle)


class BlurLoader:
    """按 epoch 动态调整模糊强度的 DataLoader 包装器"""
    def __init__(self, color_idxs, count_idxs, img_size, schedule):
        """
        schedule: [(epoch_end, blur_strength), ...]
          例如: [(15, 'strong'), (25, 'medium'), (40, None)]
        """
        self.color_idxs = color_idxs
        self.count_idxs = count_idxs
        self.img_size = img_size
        self.schedule = schedule
        self.current_epoch = 0
    
    def get_loader(self, epoch):
        self.current_epoch = epoch
        for end_ep, strength in self.schedule:
            if epoch < end_ep:
                return build_loader(self.color_idxs, self.count_idxs, 
                                    self.img_size, shuffle=True, blur_strength=strength)
        return build_loader(self.color_idxs, self.count_idxs, 
                           self.img_size, shuffle=True)


# ==================== 模型 ====================

def cbr(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c,out_c,3,padding=1),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class Tiny16(nn.Module):
    def __init__(self, nc, ich=3):
        super().__init__()
        self.net = nn.Sequential(
            cbr(ich,16), nn.MaxPool2d(2),
            cbr(16,32), nn.MaxPool2d(2),
            cbr(32,64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64,nc))
    def forward(self,x): return self.net(x)
    def features(self,x):
        for m in list(self.net.children())[:-1]: x=m(x)
        return x.flatten(1)

class Small64(nn.Module):
    def __init__(self, nc, ich=3):
        super().__init__()
        self.net = nn.Sequential(
            cbr(ich,16), nn.MaxPool2d(2),
            cbr(16,32), nn.MaxPool2d(2),
            cbr(32,64), nn.MaxPool2d(2),
            cbr(64,128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,nc))
    def forward(self,x): return self.net(x)
    def features(self,x):
        for m in list(self.net.children())[:-1]: x=m(x)
        return x.flatten(1)

class Multi64(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            cbr(3,16), nn.MaxPool2d(2),
            cbr(16,32), nn.MaxPool2d(2),
            cbr(32,64), nn.MaxPool2d(2),
            cbr(64,128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.clr_head = nn.Linear(128, N_COLORS)
        self.cnt_head = nn.Linear(128, N_COUNTS)
    def forward(self,x):
        f = self.body(x)
        return self.clr_head(f), self.cnt_head(f)
    def features(self,x): return self.body(x)


# ==================== 训练 ====================

def train_one(model, tr_ld, vl_ld, target, epochs, lr):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_st = 0, None
    for _ in range(epochs):
        model.train()
        for im, c, cnt in tr_ld:
            im = im.to(DEVICE)
            lbl = (c if target=='color' else cnt).to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(im),lbl).backward(); opt.step()
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for im, c, cnt in vl_ld:
                im = im.to(DEVICE)
                lbl = c if target=='color' else cnt
                cor += (model(im).argmax(1).cpu()==lbl).sum().item(); tot += len(lbl)
        acc = cor/tot
        if acc >= best_acc:
            best_acc=acc; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model, best_acc

def train_one_progressive(model, blur_loader, vl_ld, target, epochs, lr):
    """渐进式模糊训练: 每 epoch 获取不同模糊度的数据"""
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_st = 0, None
    for ep in range(epochs):
        tr_ld = blur_loader.get_loader(ep)
        model.train()
        for im, c, cnt in tr_ld:
            im = im.to(DEVICE)
            lbl = (c if target=='color' else cnt).to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(im),lbl).backward(); opt.step()
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for im, c, cnt in vl_ld:
                im = im.to(DEVICE)
                lbl = c if target=='color' else cnt
                cor += (model(im).argmax(1).cpu()==lbl).sum().item(); tot += len(lbl)
        acc = cor/tot
        if acc >= best_acc:
            best_acc=acc; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model, best_acc


def train_multi(model, tr_ld, vl_ld, epochs, lr):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_sum, best_st = 0, None
    for _ in range(epochs):
        model.train()
        for im, c, cnt in tr_ld:
            im, c, cnt = im.to(DEVICE), c.to(DEVICE), cnt.to(DEVICE)
            opt.zero_grad()
            co, cto = model(im)
            loss = nn.CrossEntropyLoss()(co,c) + nn.CrossEntropyLoss()(cto,cnt)
            loss.backward(); opt.step()
        model.eval()
        cc, ccnt, tot = 0, 0, 0
        with torch.no_grad():
            for im, c, cnt in vl_ld:
                im = im.to(DEVICE)
                co, cto = model(im)
                cc += (co.argmax(1).cpu()==c).sum().item()
                ccnt += (cto.argmax(1).cpu()==cnt).sum().item(); tot += len(c)
        s = cc/tot+ccnt/tot
        if s>=best_sum: best_sum=s; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model, cc/tot, ccnt/tot


# ==================== 评估 ====================

@torch.no_grad()
def ev(model, ld, task):
    model.eval(); cor, tot = 0, 0
    for im, c, cnt in ld:
        im = im.to(DEVICE)
        lbl = c if task=='color' else cnt
        cor += (model(im).argmax(1).cpu()==lbl).sum().item(); tot += len(lbl)
    return cor/tot

@torch.no_grad()
def ev_multi(model, ld):
    model.eval(); cc, ccnt, tot = 0, 0, 0
    for im, c, cnt in ld:
        im = im.to(DEVICE)
        co, cto = model(im)
        cc += (co.argmax(1).cpu()==c).sum().item()
        ccnt += (cto.argmax(1).cpu()==cnt).sum().item(); tot += len(c)
    return cc/tot, ccnt/tot

@torch.no_grad()
def extract_feats(model, ld):
    model.eval(); fs, cs = [], []
    for im, c, _ in ld:
        fs.append(model.features(im.to(DEVICE)).cpu()); cs.append(c)
    return torch.cat(fs).numpy(), torch.cat(cs).numpy()

def probe_test(feats, labels, nc):
    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    ntr = int(len(X)*0.8)
    idx = np.random.permutation(len(X))
    Xt,yt=X[idx[:ntr]],y[idx[:ntr]]; Xv,yv=X[idx[ntr:]],y[idx[ntr:]]
    p = nn.Linear(feats.shape[1], nc)
    opt = optim.Adam(p.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(); nn.CrossEntropyLoss()(p(Xt),yt).backward(); opt.step()
    with torch.no_grad(): return (p(Xv).argmax(1)==yv).float().mean().item()


# ==================== main ====================

def main():
    rbl_c = 1/N_COLORS
    rbl = 1/N_COUNTS
    
    print(f"\n配置: 3色 5类计数, 100%捷径, 训练{N_TRAIN*3}张, epoch{EPOCHS}")
    print(f"捷径: 红→3点 蓝→4点 绿→5点  随机基线={rbl:.3f}")
    
    print("\n[1] 生成数据...")
    tr_c, tr_cnt = make_dataset(N_TRAIN, SPURIOUS_RATIO)
    tr_ub_c, tr_ub_cnt = make_dataset(N_TRAIN, 0.0)
    vl_c, vl_cnt = make_counterfactual(20)
    ts_c, ts_cnt = make_counterfactual(N_CF)
    print(f"  有偏训练={len(tr_c)}  无偏训练={len(tr_ub_c)}  验证={len(vl_c)}  测试={len(ts_c)}")
    
    trL16 = build_loader(tr_c, tr_cnt, 16)
    trL64 = build_loader(tr_c, tr_cnt, 64)
    trL64g = build_loader(tr_c, tr_cnt, 64, gray=True)
    trL64ub = build_loader(tr_ub_c, tr_ub_cnt, 64)
    # 模糊变体
    trL64_blurF = build_loader(tr_c, tr_cnt, 64, blur_strength='medium')  # 固定中度模糊
    trL64_blurG = BlurLoader(tr_c, tr_cnt, 64, 
        [(20, 'strong'), (30, 'medium'), (40, None)])  # 渐进式
    
    vL16 = build_loader(vl_c, vl_cnt, 16, shuffle=False)
    vL64 = build_loader(vl_c, vl_cnt, 64, shuffle=False)
    vL64g = build_loader(vl_c, vl_cnt, 64, gray=True, shuffle=False)
    tL16 = build_loader(ts_c, ts_cnt, 16, shuffle=False)
    tL64 = build_loader(ts_c, ts_cnt, 64, shuffle=False)
    tL64g = build_loader(ts_c, ts_cnt, 64, gray=True, shuffle=False)
    
    print("\n[2] 训练模型...")
    
    mA, aA = train_one(Tiny16(N_COLORS), trL16, vL16, 'color', EPOCHS, LR)
    print(f"  [A] 16x16颜色: val={aA:.4f}")
    
    mB, aB = train_one(Small64(N_COUNTS), trL64, vL64, 'count', EPOCHS, LR)
    print(f"  [B] 64x64计数(有偏彩色): val={aB:.4f}")
    
    mC, aC = train_one(Small64(N_COUNTS,ich=1), trL64g, vL64g, 'count', EPOCHS, LR)
    print(f"  [C] 64x64灰度计数: val={aC:.4f}")
    
    mD, aDc, aDcnt = train_multi(Multi64(), trL64, vL64, EPOCHS, LR)
    print(f"  [D] 64x64多任务: 色={aDc:.4f} 计数={aDcnt:.4f}")
    
    mE, aE = train_one(Small64(N_COUNTS), trL64ub, vL64, 'count', EPOCHS, LR)
    print(f"  [E] 64x64计数(无偏/上界): val={aE:.4f}")
    
    mF, aF = train_one(Small64(N_COUNTS), trL64_blurF, vL64, 'count', EPOCHS, LR)
    print(f"  [F] 64x64固定模糊(8x8): val={aF:.4f} [新]")
    
    mG, aG = train_one_progressive(Small64(N_COUNTS), trL64_blurG, vL64, 'count', EPOCHS, LR)
    print(f"  [G] 64x64渐进模糊(4→8→原): val={aG:.4f} [新]")
    
    # ============ 测试 ============
    print("\n" + "="*65)
    print("[3] 反事实测试 (捷径已打破)")
    print("="*65)
    
    trL64ts = build_loader(tr_c, tr_cnt, 64, shuffle=False)
    trL64gts = build_loader(tr_c, tr_cnt, 64, gray=True, shuffle=False)
    trFts = build_loader(tr_c, tr_cnt, 64, shuffle=False, blur_strength='medium')
    
    tr_rB = ev(mB, trL64ts, 'count')
    tr_rC = ev(mC, trL64gts, 'count')
    _, tr_rDcnt = ev_multi(mD, trL64ts)
    tr_rF = ev(mF, trFts, 'count')
    rB = ev(mB, tL64, 'count')
    rC = ev(mC, tL64g, 'count')
    rDc, rDcnt = ev_multi(mD, tL64)
    rE = ev(mE, tL64, 'count')
    rF = ev(mF, tL64, 'count')
    rG = ev(mG, tL64, 'count')
    
    print(f"  {'模型':<25}{'训练(有偏)':>10}{'反事实':>10}{'捷径差距':>10}")
    print(f"  {'-'*57}")
    print(f"  {'B) 计数(有偏彩色)':<25}{tr_rB:>10.4f}{rB:>10.4f}{tr_rB-rB:>10.4f}")
    print(f"  {'C) 灰度计数':<25}{tr_rC:>10.4f}{rC:>10.4f}{tr_rC-rC:>10.4f}")
    print(f"  {'D) 多任务':<25}{tr_rDcnt:>10.4f}{rDcnt:>10.4f}{tr_rDcnt-rDcnt:>10.4f}")
    print(f"  {'F) 固定模糊(8x8) [新]':<25}{tr_rF:>10.4f}{rF:>10.4f}{tr_rF-rF:>10.4f}")
    print(f"  {'G) 渐进模糊 [新]':<25}{'--':>10}{rG:>10.4f}{'--':>10}")
    print(f"  {'E) 无偏(上界)':<25}{'--':>10}{rE:>10.4f}{'--':>10}")
    print(f"  {'随机基线':<25}{'--':>10}{rbl:>10.4f}{'--':>10}")
    
    # ============ 探针 ============
    print("\n[4] 探针: 模型内部是否隐式编码颜色?")
    fb, lb = extract_feats(mB, trL64ts); pb = probe_test(fb, lb, N_COLORS)
    fc, lc = extract_feats(mC, trL64gts); pc = probe_test(fc, lc, N_COLORS)
    fd, ld = extract_feats(mD, trL64ts); pd = probe_test(fd, ld, N_COLORS)
    ff, lf = extract_feats(mF, trL64ts); pf = probe_test(ff, lf, N_COLORS)
    fg, lg = extract_feats(mG, trL64ts); pg = probe_test(fg, lg, N_COLORS)
    
    print(f"  (基线{rbl_c:.3f}, >{rbl_c*1.5:.3f}=隐式颜色)")
    for name, probe in [("B 有偏彩色",pb),("C 灰度",pc),("D 多任务",pd),
                         ("F 固定模糊",pf),("G 渐进模糊",pg)]:
        print(f"  {name:<12}: {probe:.4f}  {'[!]隐式颜色' if probe>rbl_c*1.5 else '[OK]'}")
    
    # ============ 结论 ============
    print("\n" + "="*65)
    print("[5] 结论")
    print("="*65)
    
    print(f"""
  核心对比 (反事实准确率, 越高=越不受捷径影响):

  模型                    反事实    捷径差距   颜色探针
  {'-'*55}
  B) 有偏彩色(基线)     {rB:.4f}     {tr_rB-rB:.4f}      {pb:.4f}
  C) 灰度剥离           {rC:.4f}     {tr_rC-rC:.4f}      {pc:.4f}
  F) 固定模糊 [新]      {rF:.4f}     {tr_rF-rF:.4f}      {pf:.4f}
  G) 渐进模糊 [新]      {rG:.4f}     --        {pg:.4f}
  E) 无偏(上界)         {rE:.4f}     --        --
  随机基线               {rbl:.4f}
""")
    
    print("  关键发现:")
    
    if rF > rB * 1.3:
        print(f"  [V] 固定模糊(F)={rF:.3f} >> 有偏彩色(B)={rB:.3f}")
        print(f"      全局模糊大幅削弱了捷径依赖")
    elif rF > rB * 1.1:
        print(f"  [~] 固定模糊(F)={rF:.3f} > 有偏彩色(B)={rB:.3f}")
        print(f"      全局模糊有一定削弱效果")
    else:
        print(f"  [X] 固定模糊(F)={rF:.3f} vs 有偏彩色(B)={rB:.3f}")
        print(f"      固定模糊未能显著削弱捷径")
    
    if rG > rF * 1.05:
        print(f"  [V] 渐进模糊(G)={rG:.3f} > 固定模糊(F)={rF:.3f}")
        print(f"      渐进式(模拟发育)优于固定模糊")
    
    if rG > rB * 1.5:
        print(f"  [V] 渐进模糊(G)={rG:.3f} >> 有偏彩色(B)={rB:.3f}")
        print(f"      模拟婴儿感官发育策略显著有效!")

    if rC > rF:
        print(f"  [对比] 灰度剥离(C)={rC:.3f} vs 固定模糊(F)={rF:.3f}")
        print(f"         针对性通道剥离 > 全局无差别模糊")
    
    print(f"\n  全局模糊化策略判断:")
    if rG > rB * 1.3 or rF > rB * 1.3:
        print(f"  [V] '婴儿感官模拟'方向值得继续深挖")
        print(f"  [V] 模糊信息确实能帮助模型学到更干净的规律")
    else:
        print(f"  [~] 全局模糊化在本场景效果有限")
        print(f"  [~] 可能需要更精细的模糊策略或与其他手段结合")

    print("\n实验完成。")


if __name__=='__main__':
    main()
