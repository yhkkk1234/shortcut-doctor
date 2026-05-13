"""
验证模块: 评估去偏治疗效果。

在干净(无捷径)测试集上对比治疗前后的准确率,
输出治疗效果报告。
"""

import torch
import copy


class Verifier:
    """
    验证器: 对比治疗前后的模型在干净测试集上的表现。
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def evaluate(self, model, dataloader):
        """评估模型准确率"""
        model.eval()
        model.to(self.device)
        correct, total = 0, 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch, None
            
            images = images.to(self.device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            
            if labels is not None:
                correct += (preds == labels).sum().item()
                total += len(labels)
            else:
                total += len(preds)
        
        return correct / total if total > 0 else 0.0
    
    def verify(self, treated_model, clean_test_loader, 
               baseline_model=None, shortcut_test_loader=None):
        """
        验证治疗效果。
        
        Returns:
          {
            'clean_accuracy': 治疗后在干净测试集上的准确率,
            'baseline_clean_accuracy': 治疗前在干净测试集上的准确率,
            'shortcut_dependency_before': 治疗前捷径依赖度,
            'shortcut_dependency_after': 治疗后捷径依赖度,
            'improvement': 改善幅度,
            'shortcut_reduction': 捷径依赖降低幅度
          }
        """
        result = {}
        
        # 治疗后在干净测试集上的准确率
        result['clean_accuracy'] = self.evaluate(treated_model, clean_test_loader)
        
        if baseline_model is not None:
            result['baseline_clean_accuracy'] = self.evaluate(baseline_model, clean_test_loader)
            result['improvement'] = result['clean_accuracy'] - result['baseline_clean_accuracy']
        
        if shortcut_test_loader is not None:
            acc_shortcut = self.evaluate(treated_model, shortcut_test_loader)
            result['shortcut_dependency_after'] = acc_shortcut - result['clean_accuracy']
            
            if baseline_model is not None:
                acc_s_before = self.evaluate(baseline_model, shortcut_test_loader)
                result['shortcut_dependency_before'] = acc_s_before - result['baseline_clean_accuracy']
                result['shortcut_reduction'] = (
                    result['shortcut_dependency_before'] - result['shortcut_dependency_after']
                )
        
        return result
    
    def report(self, result, prescription=None):
        """生成验证报告"""
        lines = ["治疗方案验证报告", "=" * 50]
        
        if prescription:
            lines.append(f"  处方: {', '.join(prescription.to_list())}")
        
        lines.append(f"  治疗后干净集准确率: {result.get('clean_accuracy', 'N/A'):.4f}")
        
        if 'baseline_clean_accuracy' in result:
            lines.append(f"  治疗前干净集准确率: {result['baseline_clean_accuracy']:.4f}")
            imp = result.get('improvement', 0)
            lines.append(f"  改善幅度: {imp:+.4f} ({imp*100:+.1f}%)")
        
        if 'shortcut_dependency_before' in result:
            lines.append(f"  捷径依赖(治疗前): {result['shortcut_dependency_before']:.4f}")
            lines.append(f"  捷径依赖(治疗后): {result['shortcut_dependency_after']:.4f}")
            red = result.get('shortcut_reduction', 0)
            lines.append(f"  捷径依赖降低: {red:+.4f}")
        
        return "\n".join(lines)
