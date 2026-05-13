"""
============================================================================
Shortcut Doctor - 捷径诊断与去偏工具
============================================================================
核心方法论:
  1. 探针分析 → 发现模型学了什么捷径
  2. 捷径分类 → 判断捷径在高/低频哪个频道
  3. 对症下药 → 推荐对应的信息带宽控制手段
  4. 效果验证 → 在干净测试集上确认去偏效果

作者: 基于"多尺度信息带宽控制"项目实验总结
============================================================================
"""

from .probe import ProbeAnalyzer
from .diagnose import ShortcutDiagnoser, ShortcutType
from .prescribe import Prescriber, Prescription
from .treat import TreatmentApplier
from .verify import Verifier

__version__ = "0.1.0"
__all__ = [
    "ProbeAnalyzer", "ShortcutDiagnoser", "ShortcutType",
    "Prescriber", "Prescription", "TreatmentApplier", "Verifier"
]
