"""
治疗模块: 应用信息带宽控制手段。

支持的去偏手段:
  - grayscale:          剥离颜色通道
  - progressive_blur:   渐进式模糊 (模拟发育)
  - fixed_blur:         固定模糊
  - downscale:          降分辨率
  - channel_drop:       随机丢弃通道
"""

import torch
import torch.nn.functional as F


class TreatmentApplier:
    """
    去偏治疗器: 将处方中的手段应用到训练流程中。
    
    用法:
        applier = TreatmentApplier(prescription)
        for epoch in range(epochs):
            for batch in dataloader:
                batch = applier.transform(batch, epoch, total_epochs)
                output = model(batch)
    """
    
    def __init__(self, prescription, img_size=32):
        self.prescription = prescription
        self.img_size = img_size
        self.methods = {m[0]: m[1] for m in prescription.methods}
    
    def transform(self, images, epoch=None, total_epochs=None):
        """
        对一批图像应用去偏变换。
        
        images: (B, C, H, W) tensor
        epoch / total_epochs: 用于渐进模糊的调度
        """
        result = images
        
        for method_name in self.methods:
            if method_name == 'grayscale' and images.shape[1] == 3:
                result = self._apply_grayscale(result)
            
            elif method_name == 'progressive_blur' and epoch is not None:
                strength = self._blur_schedule(epoch, total_epochs)
                result = self._apply_blur(result, strength)
            
            elif method_name == 'fixed_blur':
                result = self._apply_blur(result, 'medium')
            
            elif method_name == 'downscale':
                result = self._apply_downscale(result)
        
        return result
    
    def _apply_grayscale(self, images):
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.expand(-1, 3, -1, -1)
    
    def _apply_blur(self, images, strength):
        size_map = {'strong': 4, 'medium': 8, 'mild': 16, 'light': 24}
        target = size_map.get(strength, 8)
        orig_h, orig_w = images.shape[-2:]
        small = F.interpolate(images, size=(target, target), mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    def _apply_downscale(self, images):
        target = max(8, self.img_size // 2)
        return F.interpolate(images, size=(target, target), mode='bilinear', align_corners=False)
    
    def _blur_schedule(self, epoch, total_epochs):
        """渐进模糊调度: 早期强模糊, 后期无模糊"""
        if total_epochs is None:
            return 'medium'
        progress = epoch / total_epochs
        if progress < 0.4:
            return 'strong'
        elif progress < 0.6:
            return 'medium'
        elif progress < 0.85:
            return 'mild'
        return None  # 不模糊
    
    def transform_dataloader(self, dataloader, epoch, total_epochs):
        """生成器: 逐batch应用变换"""
        for images, labels in dataloader:
            yield self.transform(images, epoch, total_epochs), labels
    
    def report(self):
        lines = ["应用的治疗手段:", "-" * 30]
        for name, strength in self.methods.items():
            lines.append(f"  {name} (强度: {strength})")
        return "\n".join(lines)
