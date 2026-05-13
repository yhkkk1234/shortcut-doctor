# Shortcut Doctor

捷径诊断与去偏工具 —— 基于"信息带宽控制"方法论。

**核心理念:** 模型学坏的根源是信息太多。不是改模型架构，而是**在信息入口处做减法**。

## 四步工作流

```
探针分析 → 捷径诊断 → 对症处方 → 治疗验证
(probe)    (diagnose)  (prescribe)  (treat+verify)
```

## 快速开始

```bash
pip install -r requirements.txt
python demo.py
```

## API 用法

```python
from shortcut_doctor import (
    ProbeAnalyzer, ShortcutDiagnoser, ShortcutType,
    Prescriber, TreatmentApplier, Verifier
)

# 1. 探针分析 - 模型内部偷偷学了什么?
analyzer = ProbeAnalyzer(trained_model)
analyzer.extract_features(dataloader)
analyzer.add_suspect('color', color_labels)     # 你怀疑的捷径
analyzer.add_suspect('texture', texture_labels)
results = analyzer.test_all_suspects({'color': 3, 'texture': 2})

# 2. 捷径诊断 - 捷径在哪个频段?
diagnoser = ShortcutDiagnoser()
diagnosis = diagnoser.diagnose(results, {'color': ShortcutType.COLOR})

# 3. 对症处方 - 低频捷径→灰度, 高频捷径→模糊
prescriber = Prescriber()
prescription = prescriber.prescribe(diagnosis)

# 4. 应用治疗 + 验证
applier = TreatmentApplier(prescription)
for epoch in range(epochs):
    for images, labels in dataloader:
        images = applier.transform(images, epoch, epochs)
        output = model(images)
        loss.backward()

verifier = Verifier()
result = verifier.verify(treated_model, clean_test_loader, baseline_model)
```

## 支持的治疗手段

| 手段 | 针对捷径 | 机制 |
|------|---------|------|
| `grayscale` | 颜色偏置 | 物理剥离颜色通道 |
| `progressive_blur` | 纹理/高频偏置 | 渐进式模糊(模拟发育) |
| `fixed_blur` | 高频偏置 | 固定模糊度 |
| `downscale` | 形状/位置偏置 | 降分辨率 |

## 决策树

```
捷径是颜色偏置?  → 灰度化
捷径是纹理偏置?  → 渐进模糊
捷径是形状偏置?  → 降分辨率
捷径是混合型?    → 灰度 + 模糊
捷径未知?        → 渐进模糊(保守) + 灰度(可选)
```

## 实验证据

本工具基于三个独立实验的完整验证:

| 实验 | 数据 | 捷径 | 有效手段 | 效果 |
|------|------|------|---------|------|
| 合成圆点 | 自制 | 颜色→点数 | 灰度 | +133% |
| 合成图案 | 自制 | 色块→类别 | 灰度 | +∞ (依赖消除) |
| CIFAR-100 | 真实 | 色边框→类别 | 灰度 | 待验证 |

## 引用

如果此工具对你有帮助, 可引用:

```
@misc{shortcut-doctor,
  title  = {Shortcut Doctor: Information Bandwidth Control for Debiasing},
  author = {基于"多尺度信息带宽控制视觉识别"项目},
  year   = {2026},
  note   = {https://github.com/...}
}
```

## 许可

MIT License
