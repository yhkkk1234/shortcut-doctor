"""
=============================================================================
仿生输入层 v3 — 对抗式通道混合
=============================================================================
验证: 通过对抗颜色分类器, 迫使仿生层学会主动抑制颜色捷径
=============================================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random, sys
from contextlib import nullcontext

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
    DEVICE = torch_directml.device(); print(f"Device: DirectML")
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Device: {DEVICE}")


class BionicInputLayer(nn.Module):
    """
    仿生输入层 v3: 可学习通道混合 + 默认灰度初始化
    
    关键改动: 矩阵初始化为全1/3 (强制灰度), 模型必须主动恢复颜色
    原理: "opt-in color" — 默认给灰度, 模型有需要才花参数恢复
    """
    def __init__(self, in_channels=3, kernel_size=5, init_sigma=2.0):
        super().__init__()
        # 默认灰度化!
        self.color_matrix = nn.Parameter(torch.ones(in_channels, in_channels) / 3)
        self.log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma)))
        self.channel_gain = nn.Parameter(torch.ones(in_channels))
        k = kernel_size
        self.spatial_conv = nn.Conv2d(1, 1, k, padding=k//2, bias=False)
        center = k//2
        xs = torch.arange(k, dtype=torch.float32)-center
        ys = torch.arange(k, dtype=torch.float32)-center
        g = torch.exp(-(xs.unsqueeze(0)**2+ys.unsqueeze(1)**2)/(2*init_sigma**2))
        with torch.no_grad(): self.spatial_conv.weight.data[0,0]=g/g.sum()
    
    def forward(self, x):
        x = torch.einsum('bchw,cd->bdhw', x, self.color_matrix)
        x = x * self.channel_gain.view(1,-1,1,1)
        B,C,H,W = x.shape
        x_flat = x.reshape(B*C,1,H,W)
        return self.spatial_conv(x_flat).reshape(B,C,H,W)
    
    @property
    def sigma(self): return torch.exp(self.log_sigma)
    
    def get_mixing_strength(self):
        """返回通道混合偏离灰度矩阵的程度"""
        gray = torch.ones(3, 3) / 3
        return (self.color_matrix.data.cpu() - gray).abs().mean().item()


class ColorAdversary(nn.Module):
    """微型颜色分类对抗器: 2层MLP"""
    def __init__(self, in_channels=3, n_colors=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, 16), nn.ReLU(),
            nn.Linear(16, n_colors))
    def forward(self, x): return self.net(x)


# ==================== 对抗训练 ====================

def cbr(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c,out_c,3,padding=1),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class MainNet(nn.Module):
    def __init__(self, nc, ich=3):
        super().__init__()
        self.net = nn.Sequential(
            cbr(ich,16), nn.MaxPool2d(2), cbr(16,32), nn.MaxPool2d(2),
            cbr(32,64), nn.MaxPool2d(2), cbr(64,128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128,nc))
    def forward(self, x): return self.net(x)


def train_adversarial(bionic, main_net, adversary, tr_ld, vl_ld, epochs, lr, 
                       lambda_adv=0.5):
    """
    对抗训练:
      - 主网络: min count_loss (正常)
      - 对抗器: min color_loss (正常训练)
      - 仿生层: min count_loss - lambda * color_loss (反向颜色梯度)
    
    效果: 仿生层学会"压颜色=帮主任务但不帮对抗器"
    """
    bionic = bionic.to(DEVICE)
    main_net = main_net.to(DEVICE)
    
    if adversary is None:
        # Plain mode: bionic + main, no adversary
        opt_bionic = optim.AdamW(bionic.parameters(), lr=lr*0.5)
        opt_main = optim.AdamW(main_net.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        best_acc, best_st = 0, None
        for ep in range(epochs):
            bionic.train(); main_net.train()
            for im, c, cnt in tr_ld:
                im, cnt_lbl = im.to(DEVICE), cnt.to(DEVICE)
                opt_bionic.zero_grad(); opt_main.zero_grad()
                crit(main_net(bionic(im)), cnt_lbl).backward()
                opt_bionic.step(); opt_main.step()
            bionic.eval(); main_net.eval(); cor, tot = 0, 0
            with torch.no_grad():
                for im, _, cnt in vl_ld:
                    im = im.to(DEVICE)
                    cor += (main_net(bionic(im)).argmax(1).cpu()==cnt).sum().item()
                    tot += len(cnt)
            acc = cor/tot
            if acc >= best_acc:
                best_acc = acc
                best_st = {'bionic': {k:v.cpu().clone() for k,v in bionic.state_dict().items()},
                          'main': {k:v.cpu().clone() for k,v in main_net.state_dict().items()}}
        bionic.load_state_dict(best_st['bionic']); main_net.load_state_dict(best_st['main'])
        return bionic, main_net, best_acc
    
    adversary = adversary.to(DEVICE)
    
    # 仿生层独立优化器
    opt_bionic = optim.AdamW(bionic.parameters(), lr=lr*0.5)
    opt_main = optim.AdamW(main_net.parameters(), lr=lr)
    opt_adv = optim.AdamW(adversary.parameters(), lr=lr*2)
    
    crit = nn.CrossEntropyLoss()
    best_acc, best_st = 0, None
    
    for ep in range(epochs):
        bionic.train(); main_net.train(); adversary.train()
        
        for im, c, cnt in tr_ld:
            im, c_lbl, cnt_lbl = im.to(DEVICE), c.to(DEVICE), cnt.to(DEVICE)
            
            # 前向
            bionic_out = bionic(im)
            count_out = main_net(bionic_out)
            color_out = adversary(bionic_out.detach())  # detach: 对抗器梯度不回溯
            count_loss = crit(count_out, cnt_lbl)
            color_loss = crit(color_out, c_lbl)
            
            # 仿生层: 鼓励主任务, 惩罚颜色
            opt_bionic.zero_grad()
            (count_loss - lambda_adv * color_loss).backward(retain_graph=True)
            opt_bionic.step()
            
            # 主网络: 只关心计数
            opt_main.zero_grad()
            opt_adv.zero_grad()
            
            bionic_out2 = bionic(im)  # 重新过仿生层 (参数已更新)
            count_out2 = main_net(bionic_out2)
            crit(count_out2, cnt_lbl).backward()
            opt_main.step()
            
            # 对抗器: 只关心颜色分类
            with torch.no_grad():
                bionic_out3 = bionic(im)
            color_out3 = adversary(bionic_out3)
            crit(color_out3, c_lbl).backward()
            opt_adv.step()
        
        # 验证
        bionic.eval(); main_net.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for im, _, cnt in vl_ld:
                im = im.to(DEVICE)
                out = main_net(bionic(im))
                cor += (out.argmax(1).cpu()==cnt).sum().item()
                tot += len(cnt)
        acc = cor/tot
        if acc >= best_acc:
            best_acc = acc
            best_st = {'bionic': {k:v.cpu().clone() for k,v in bionic.state_dict().items()},
                      'main': {k:v.cpu().clone() for k,v in main_net.state_dict().items()}}
    
    bionic.load_state_dict(best_st['bionic'])
    main_net.load_state_dict(best_st['main'])
    return bionic, main_net, best_acc


# ==================== 数据 ====================

def generate_batch(color_idxs, dot_counts):
    B = len(color_idxs); H=W=BASE_SIZE
    imgs = np.ones((B,3,H,W), dtype=np.float32)
    ctr,rad=H//2,H//3
    yg,xg=np.ogrid[:H,:W]; dist=np.sqrt((xg-ctr)**2+(yg-ctr)**2)
    fm,rm=dist<=rad,(dist<=rad)&(dist>=rad-3)
    for b in range(B):
        rgb=np.array(COLORS[color_idxs[b]],dtype=np.float32)/255.0
        fc=rgb*0.25+0.75
        for ch in range(3): imgs[b,ch][fm]=fc[ch]; imgs[b,ch][rm]=rgb[ch]
        dc=dot_counts[b]; nd=(dc.item() if hasattr(dc,'item') else dc)+1
        mo=rad-DOT_R-6
        for d in range(nd):
            a=2*np.pi*d/nd+random.uniform(-0.15,0.15) if nd>1 else 0
            o=mo*random.uniform(0.30,0.88) if nd>1 else 0
            px,py=ctr+int(o*np.cos(a)),ctr+int(o*np.sin(a))
            dy,dx=np.ogrid[:H,:W]
            imgs[b,:][:,np.sqrt((dx-px)**2+(dy-py)**2)<=DOT_R]=0.05
    return torch.from_numpy(imgs).float()

def make_dataset(n, sr):
    cd, ctd = [], []
    for ci in range(N_COLORS):
        for _ in range(n):
            cd.append(ci)
            ctd.append(SPURIOUS_MAP[ci] if random.random()<sr else random.randint(0,N_COUNTS-1))
    return cd,ctd

def make_cf(n):
    cd, ctd = [], []
    for ci in range(N_COLORS):
        for cnt in range(N_COUNTS):
            for _ in range(n): cd.append(ci); ctd.append(cnt)
    return cd,ctd

def build_loader(cd,ctd,sz,gray=False,shuf=True):
    imgs=[]
    for s in range(0,len(cd),256):
        e=min(s+256,len(cd))
        b=generate_batch(cd[s:e],ctd[s:e])
        b=F.interpolate(b,(sz,sz),mode='bilinear',align_corners=False)
        if gray: b=b.mean(1,keepdim=True)
        imgs.append(b)
    im=torch.cat(imgs)
    return DataLoader(TensorDataset(im,torch.tensor(cd,dtype=torch.long),torch.tensor(ctd,dtype=torch.long)),BS,shuffle=shuf)

def train_simple(model, tr_ld, vl_ld, epochs, lr):
    model=model.to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    ba,bs=0,None
    for _ in range(epochs):
        model.train()
        for im,_,cnt in tr_ld:
            im,cnt=im.to(DEVICE),cnt.to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(im),cnt).backward(); opt.step()
        model.eval(); c,t=0,0
        with torch.no_grad():
            for im,_,cnt in vl_ld:
                im=im.to(DEVICE); c+=(model(im).argmax(1).cpu()==cnt).sum().item(); t+=len(cnt)
        a=c/t
        if a>=ba: ba=a; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(bs)
    return model,ba

@torch.no_grad()
def ev(model, ld):
    model.eval(); c,t=0,0
    for im,_,cnt in ld:
        im=im.to(DEVICE); c+=(model(im).argmax(1).cpu()==cnt).sum().item(); t+=len(cnt)
    return c/t

@torch.no_grad()
def ev_bionic(bionic, main, ld):
    """评估仿生模型: 先过仿生层再过主网络"""
    bionic.eval(); main.eval(); c,t=0,0
    for im,_,cnt in ld:
        im=im.to(DEVICE)
        c+=(main(bionic(im)).argmax(1).cpu()==cnt).sum().item(); t+=len(cnt)
    return c/t


# ==================== main ====================

def main():
    rbl=1/N_COUNTS
    print(f"\nFz vs lb | shortcut 100% | {N_TRAIN*3} train | {EPOCHS}ep")
    
    tr_c,tr_cnt=make_dataset(N_TRAIN,SPURIOUS_RATIO)
    vl_c,vl_cnt=make_cf(20); ts_c,ts_cnt=make_cf(N_CF)
    
    trL=build_loader(tr_c,tr_cnt,64); trLg=build_loader(tr_c,tr_cnt,64,gray=True)
    vL=build_loader(vl_c,vl_cnt,64,shuf=False); vLg=build_loader(vl_c,vl_cnt,64,gray=True,shuf=False)
    tL=build_loader(ts_c,ts_cnt,64,shuf=False); tLg=build_loader(ts_c,ts_cnt,64,gray=True,shuf=False)
    
    print("\n[1] Training...")
    
    # H) Baseline
    mH={}; mH['main'],aH=train_simple(MainNet(N_COUNTS),trL,vL,EPOCHS,LR)
    rH=ev(mH['main'],tL); print(f"  [H] Baseline: val={aH:.4f}  cf={rH:.4f}")
    
    # C) Grayscale reference
    mC,_=train_simple(MainNet(N_COUNTS,ich=1),trLg,vLg,EPOCHS,LR)
    rC=ev(mC,tLg); print(f"  [C] Grayscale: cf={rC:.4f}")
    
    # M) Bionic grayscale-init, normal training (no adversary)
    bionicM = BionicInputLayer(); mainM = MainNet(N_COUNTS)
    bionicM, mainM, aM = train_adversarial(bionicM, mainM, None, trL, vL, EPOCHS, LR, lambda_adv=0.0)
    rM = ev_bionic(bionicM, mainM, tL)
    mxM = bionicM.get_mixing_strength()
    matM = bionicM.color_matrix.data.cpu().tolist()
    print(f"  [M] Grayscale-init, plain: val={aM:.4f}  cf={rM:.4f}")
    print(f"      mixing={mxM:.4f} matrix:")
    for row in matM: print(f"        [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")
    
    # N) Bionic grayscale-init + strong adversary L=3.0
    bionicN = BionicInputLayer(); mainN = MainNet(N_COUNTS); advN = ColorAdversary()
    bionicN, mainN, aN = train_adversarial(bionicN, mainN, advN, trL, vL, EPOCHS, LR, lambda_adv=3.0)
    rN = ev_bionic(bionicN, mainN, tL)
    mxN = bionicN.get_mixing_strength()
    matN = bionicN.color_matrix.data.cpu().tolist()
    print(f"  [N] Grayscale-init + adv L=3: val={aN:.4f}  cf={rN:.4f}")
    print(f"      mixing={mxN:.4f} matrix:")
    for row in matN: print(f"        [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")
    
    # Results
    print("\n"+"="*65)
    print("[2] Results")
    print("="*65)
    print(f"""
  {'Model':<28}{'CF':>8}{'vs H':>8}{'Mixing':>8}
  {'-'*54}
  H) Baseline              {rH:>8.4f}{'--':>8}  {'--':>8}
  C) Grayscale             {rC:>8.4f}{rC-rH:>+8.4f}  {'--':>8}
  M) Gray-init plain       {rM:>8.4f}{rM-rH:>+8.4f}{mxM:>8.4f}
  N) Gray-init adv L=3     {rN:>8.4f}{rN-rH:>+8.4f}{mxN:>8.4f}
  Random baseline          {rbl:>8.4f}
""")
    
    print("  Key findings:")
    if rM>rH*1.15:
        print(f"  [V] M: +{(rM-rH)*100:.1f}pp — grayscale init alone helps")
    if rN>rH*1.15:
        print(f"  [V] N: +{(rN-rH)*100:.1f}pp — adversary further boosts")
    
    if mxM>0.05:
        print(f"  [V] M mixing={mxM:.4f} — model actively deviated from grayscale")
    else:
        print(f"  [~] M mixing={mxM:.4f} — model stayed near grayscale init")
    
    if mxN>0.05:
        print(f"  [V] N mixing={mxN:.4f} — adversary pushed model away from grayscale")
    
    if mxM<0.02 and rM>rH*1.1:
        print(f"  [V] Key insight: staying grayscale (mixing=0) + higher CF acc")
        print(f"  [V] → model learned that NOT mixing = better for counting")
    
    print("\nDone.")


if __name__=='__main__':
    main()
