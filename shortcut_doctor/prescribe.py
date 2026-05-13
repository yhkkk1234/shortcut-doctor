"""
处方模块: 根据捷径类型推荐对应的信息带宽控制手段。

决策规则 (基于三实验验证):
  - COLOR / SHAPE 捷径 → 灰度化/通道剥离 (低频捷径不惧模糊)
  - TEXTURE 捷径     → 渐进模糊 (高频捷径先被模糊杀死)
  - POSITION 捷径    → 降分辨率 (位置信息在中分辨率被破坏)
  - MIXED / UNKNOWN  → 组合策略 (灰度 + 模糊)
  - 强制模糊          → 如果真规律是低频结构, 渐进模糊也可用
"""

from .diagnose import ShortcutType


class Prescription:
    """处方: 推荐的信息带宽控制手段"""
    
    def __init__(self, methods, reasoning=""):
        self.methods = methods        # [(method_name, strength, reason)]
        self.reasoning = reasoning
    
    def __repr__(self):
        m = ", ".join(f"{name}({strength})" for name, strength, _ in self.methods)
        return f"Prescription({m})"
    
    def to_list(self):
        return [name for name, _, _ in self.methods]
    
    def report(self):
        lines = ["去偏处方", "=" * 50, f"  理由: {self.reasoning}", ""]
        for name, strength, reason in self.methods:
            lines.append(f"  [{strength}] {name} — {reason}")
        return "\n".join(lines)


class Prescriber:
    """
    处方生成器: 根据捷径诊断结果, 推荐最合适的信息带宽控制手段。
    """
    
    def prescribe(self, diagnosis, frequency_profile=None):
        """
        根据诊断结果生成处方。
        
        diagnosis: ShortcutDiagnoser.diagnose() 的结果
        frequency_profile: 可选的频段分析
        """
        methods = []
        reasons = []
        
        for name, stype in diagnosis.items():
            if stype == ShortcutType.COLOR:
                methods.append(("grayscale", "强推荐", "颜色是低频捷径, 模糊无效, 需灰度化"))
                reasons.append(f"{name}是颜色捷径")
            elif stype == ShortcutType.TEXTURE:
                methods.append(("progressive_blur", "推荐", "纹理是高频捷径, 渐进模糊优先破坏"))
                reasons.append(f"{name}是纹理捷径")
            elif stype == ShortcutType.SHAPE:
                methods.append(("downscale", "推荐", "形状是低频捷径, 降分辨率有效"))
                reasons.append(f"{name}是形状捷径")
            elif stype == ShortcutType.POSITION:
                methods.append(("downscale", "推荐", "位置信息在中分辨率被破坏"))
                reasons.append(f"{name}是位置捷径")
            elif stype == ShortcutType.MIXED:
                methods.append(("grayscale", "推荐", "混合捷径至少需要剥离颜色"))
                methods.append(("progressive_blur", "可选", "叠加模糊处理高频部分"))
                reasons.append(f"{name}是混合捷径")
            else:
                # UNKNOWN: 保守策略
                methods.append(("progressive_blur", "保守推荐", "通用策略, 盲加可用"))
                methods.append(("grayscale", "可选", "如果怀疑颜色捷径"))
                reasons.append(f"{name}类型未知, 采用保守策略")
        
        # 去重
        seen = set()
        unique_methods = []
        for m in methods:
            if m[0] not in seen:
                unique_methods.append(m)
                seen.add(m[0])
        
        reasoning = "; ".join(set(reasons))
        return Prescription(unique_methods, reasoning)
    
    def decision_tree(self):
        """返回决策树的文本表示"""
        return f"""
捷径治疗决策树:
  ├── 捷径是颜色偏置?  → 灰度化 (grayscale)
  ├── 捷径是纹理偏置?  → 渐进模糊 (progressive_blur)
  ├── 捷径是形状偏置?  → 降分辨率 (downscale)
  ├── 捷径是位置偏置?  → 降分辨率 (downscale)
  ├── 捷径是混合型?    → 灰度 + 模糊 组合
  └── 捷径类型未知?    → 渐进模糊(保守) + 可叠加灰度
"""
