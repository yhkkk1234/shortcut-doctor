"""
捷径诊断模块: 根据探针结果, 分类捷径类型并给出频段分析。

捷径分类:
  - COLOR:      颜色偏置 (低频全局信号)
  - TEXTURE:    纹理/高频偏置 (高频细节信号)
  - SHAPE:      形状/轮廓偏置 (低频结构性信号)
  - POSITION:   位置偏置 (区域信号)
  - MIXED:      混合型捷径
  - UNKNOWN:    无法确定类型
"""

from enum import Enum


class ShortcutType(Enum):
    COLOR = "color"
    TEXTURE = "texture"
    SHAPE = "shape"
    POSITION = "position"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# 关键词 → 捷径类型映射
_TYPE_KEYWORDS = {
    ShortcutType.COLOR: [
        'color', 'colour', 'hue', 'rgb', 'brightness',
        'tint', 'saturation', 'chromatic', 'grayscale'
    ],
    ShortcutType.TEXTURE: [
        'texture', 'pattern', 'grain', 'noise', 'detail',
        'frequency', 'edge', 'gradient', 'watermark', 'artifact'
    ],
    ShortcutType.SHAPE: [
        'shape', 'contour', 'outline', 'silhouette', 'boundary',
        'morphology', 'geometry', 'structure', 'form'
    ],
    ShortcutType.POSITION: [
        'position', 'location', 'corner', 'center', 'border',
        'region', 'patch', 'coordinate', 'offset'
    ],
}


class ShortcutDiagnoser:
    """
    捷径诊断器: 结合探针结果和用户提供的语义信息, 判断捷径类型。
    
    判断逻辑:
      1. 用户可显式提供捷径类型 (最准确)
      2. 否则根据嫌疑人名称关键词自动匹配
      3. 无法匹配时标记为 UNKNOWN
    """
    
    def __init__(self):
        self._diagnosis = {}
    
    def diagnose(self, probe_results, suspect_descriptions=None):
        """
        对每个被探针确认的捷径进行分类。
        
        probe_results: ProbeAnalyzer.test_all_suspects() 的结果
        suspect_descriptions: {name: ShortcutType} 或 {name: str}
        """
        for name, result in probe_results.items():
            if not result['is_encoded']:
                continue
            
            # 优先用显式描述
            if suspect_descriptions and name in suspect_descriptions:
                desc = suspect_descriptions[name]
                if isinstance(desc, ShortcutType):
                    self._diagnosis[name] = desc
                else:
                    self._diagnosis[name] = self._match_keywords(str(desc))
            else:
                # 自动关键词匹配
                self._diagnosis[name] = self._match_keywords(name)
        
        return self._diagnosis
    
    def _match_keywords(self, text):
        text_lower = text.lower()
        for stype, keywords in _TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return stype
        return ShortcutType.UNKNOWN
    
    def get_frequency_profile(self):
        """
        返回捷径的频段特征, 用于匹配信息带宽控制手段。
        
        Returns:
          {
            'color':    {'freq': 'low',  'blur_resistant': True},
            'texture':  {'freq': 'high', 'blur_resistant': False},
            'shape':    {'freq': 'low',  'blur_resistant': True},
            'position': {'freq': 'mid',  'blur_resistant': False},
          }
        """
        profile = {}
        for name, stype in self._diagnosis.items():
            if stype == ShortcutType.COLOR:
                profile[name] = {'freq': 'low', 'blur_resistant': True}
            elif stype == ShortcutType.TEXTURE:
                profile[name] = {'freq': 'high', 'blur_resistant': False}
            elif stype == ShortcutType.SHAPE:
                profile[name] = {'freq': 'low', 'blur_resistant': True}
            elif stype == ShortcutType.POSITION:
                profile[name] = {'freq': 'mid', 'blur_resistant': False}
            else:
                profile[name] = {'freq': 'unknown', 'blur_resistant': None}
        return profile
    
    def report(self):
        if not self._diagnosis:
            return "未发现捷径。"
        
        lines = ["捷径诊断报告", "=" * 50]
        profile = self.get_frequency_profile()
        for name, stype in self._diagnosis.items():
            pf = profile[name]
            lines.append(
                f"  {name}: {stype.value} "
                f"(频段={pf['freq']}, 抗模糊={'是' if pf['blur_resistant'] else '否'})"
            )
        return "\n".join(lines)
