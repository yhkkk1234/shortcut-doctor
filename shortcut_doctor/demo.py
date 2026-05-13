"""
Shortcut Doctor 完整演示
=======================
演示四步法: 诊断 → 处方 → 治疗 → 验证

场景: 合成彩色圆点图 (红→3点 蓝→4点 绿→5点)
捷径: 颜色(低频) → 推荐灰度化治疗
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shortcut_doctor import (
    ProbeAnalyzer, ShortcutDiagnoser, ShortcutType,
    Prescriber, TreatmentApplier, Verifier
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ==================== 合成数据 (圆点实验复刻) ====================
N_COLORS, N_COUNTS = 3, 5
DOT_R, BASE = 6, 128
COLORS = [(220,50,50),(50,100,220),(50,180,50)]

def make_image(color_idx, n_dots):
    img = np.ones((BASE, BASE, 3), dtype=np.float32)
    ctr, rad = BASE//2, BASE//3
    yg, xg = np.ogrid[:BASE, :BASE]
    dist = np.sqrt((xg-ctr)**2+(yg-ctr)**2)
    
    rgb = np.array(COLORS[color_idx], dtype=np.float32)/255.0
    fill = rgb*0.25+0.75
    for ch in range(3):
        img[:,:,ch][dist<=rad] = fill[ch]
        img[:,:,ch][(dist<=rad)&(dist>=rad-3)] = rgb[ch]
    
    for d in range(n_dots):
        angle = 2*np.pi*d/n_dots+random.uniform(-0.15,0.15)
        off = (rad-DOT_R-6)*random.uniform(0.30,0.88)
        px = ctr+int(off*np.cos(angle))
        py = ctr+int(off*np.sin(angle))
        dy,dx=np.ogrid[:BASE,:BASE]
        dot_mask = np.sqrt((dx-px)**2+(dy-py)**2)<=DOT_R
        for ch in range(3):
            img[:,:,ch][dot_mask] = 0.05
    
    # HWC -> CHW
    img = torch.from_numpy(img.transpose(2,0,1)).float()
    img = F.interpolate(img.unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze(0)
    return img

def make_dataset(n_per_color, biased=True):
    imgs, color_lbl, count_lbl = [], [], []
    for ci in range(N_COLORS):
        for _ in range(n_per_color):
            if biased:
                cnt = [2,3,4][ci]+1  # 100%捷径
            else:
                cnt = random.randint(1, N_COUNTS)
            imgs.append(make_image(ci, cnt))
            color_lbl.append(ci)
            count_lbl.append(cnt-1)
    return TensorDataset(torch.stack(imgs), 
                         torch.tensor(color_lbl, dtype=torch.long),
                         torch.tensor(count_lbl, dtype=torch.long))

# ==================== 模型 ====================
class DemoCNN(nn.Module):
    def __init__(self, nc, ich=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ich,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,nc))
    def forward(self,x): return self.net(x)
    def features(self,x):
        for m in list(self.net.children())[:-1]: x=m(x)
        return x.flatten(1)


def train(model, loader, epochs=30, lr=3e-4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for im, _, cnt in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(im), cnt).backward()
            opt.step()

@torch.no_grad()
def acc(model, loader):
    model.eval(); c,t=0,0
    for im,_,cnt in loader:
        c+=(model(im).argmax(1)==cnt).sum().item(); t+=len(cnt)
    return c/t


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("Shortcut Doctor 完整演示")
    print("=" * 60)
    
    print("\n[准备] 生成数据 + 训练基线模型...")
    train_biased = make_dataset(500, biased=True)
    test_clean = make_dataset(40, biased=False)
    
    trL = DataLoader(train_biased, 128, shuffle=True)
    tsL = DataLoader(test_clean, 128)
    
    # 训练"吃了捷径"的模型
    model = DemoCNN(N_COUNTS)
    train(model, trL)
    baseline_acc = acc(model, tsL)
    print(f"  基线模型在干净测试集上: {baseline_acc:.4f} (随机=0.200)")
    
    # ============ Step 1: 探针分析 ============
    print("\n[步骤1] 探针分析 — 模型内部是否隐式编码了颜色?")
    
    analyzer = ProbeAnalyzer(model)
    analyzer.extract_features(trL)
    analyzer.add_suspect('color', train_biased.tensors[1])  # 颜色标签
    results = analyzer.test_all_suspects()
    print(analyzer.report())
    
    # ============ Step 2: 捷径诊断 ============
    print("\n[步骤2] 捷径诊断 — 颜色 = 低频捷径, 抗模糊")
    
    diagnoser = ShortcutDiagnoser()
    diagnosis = diagnoser.diagnose(results, {'color': ShortcutType.COLOR})
    print(diagnoser.report())
    
    # ============ Step 3: 处方 ============
    print("\n[步骤3] 生成处方 — 低频捷径 → 推荐灰度化")
    
    prescriber = Prescriber()
    prescription = prescriber.prescribe(diagnosis)
    print(prescription.report())
    
    # ============ Step 4: 治疗 ============
    print("\n[步骤4] 应用治疗 — 灰度化 + 重训练...")
    
    treated_model = DemoCNN(N_COUNTS, ich=3)  # 灰度但保留3通道
    # 灰度版本的数据
    gray_images = train_biased.tensors[0].mean(dim=1, keepdim=True).expand(-1,3,-1,-1)
    gray_ds = TensorDataset(gray_images, train_biased.tensors[1], train_biased.tensors[2])
    train(treated_model, DataLoader(gray_ds, 128, shuffle=True))
    
    # ============ Step 5: 验证 ============
    print("\n[步骤5] 验证治疗效果")
    
    verifier = Verifier()
    result = verifier.verify(treated_model, tsL, model, None)
    print(verifier.report(result, prescription))
    
    # ============ 总结 ============
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    treated_acc = acc(treated_model, tsL)
    print(f"  治疗前(彩色模型): {baseline_acc:.4f}")
    print(f"  治疗后(灰度模型): {treated_acc:.4f}")
    print(f"  改善: {treated_acc-baseline_acc:+.4f}")
    print(f"\n  处方有效性: ", end="")
    if treated_acc > baseline_acc * 1.2:
        print("[V] 治疗有效, 捷径依赖显著降低!")
    elif treated_acc > baseline_acc:
        print("[~] 有一定改善")
    else:
        print("[~] 此场景不适用 (可能是真任务也被灰度损害)")
    
    print("\n  这就是 Shortcut Doctor 的完整工作流: 探针→诊断→处方→治疗→验证")

if __name__ == '__main__':
    main()
