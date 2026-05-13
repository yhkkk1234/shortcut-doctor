"""
探针分析模块: 检测模型内部是否隐式编码了不相关但有害的特征。

核心方法:
  1. 从模型中间层提取特征
  2. 用线性探针(逻辑回归)预测"被怀疑是捷径"的标签
  3. 如果探针准确率显著高于随机基线, 说明捷径被隐式编码了
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


class ProbeAnalyzer:
    """
    线性探针分析器。
    
    用法:
        analyzer = ProbeAnalyzer(model)
        analyzer.extract_features(dataloader)   # 提取特征
        results = analyzer.test_all_suspects()  # 测试所有嫌疑人标签
    
    嫌疑人标签是用户在训练时知道但不希望模型依赖的特征,
    比如: 颜色、纹理来源、图片背景类型等。
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._features = None
        self._suspect_labels = {}
        self._probe_results = {}
    
    @torch.no_grad()
    def extract_features(self, dataloader, feature_fn=None):
        """
        从模型提取特征向量。
        
        feature_fn: 可选的函数, 接收模型和一批输入, 返回特征tensor。
                    如果为None, 尝试调用 model.features(x)。
        """
        self.model.eval()
        self.model.to(self.device)
        features = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(self.device)
            
            if feature_fn is not None:
                feats = feature_fn(self.model, inputs)
            elif hasattr(self.model, 'features'):
                feats = self.model.features(inputs)
            elif hasattr(self.model, 'get_features'):
                feats = self.model.get_features(inputs)
            else:
                raise ValueError(
                    "模型没有 features() 或 get_features() 方法, "
                    "请提供 feature_fn 参数"
                )
            
            features.append(feats.cpu())
        
        self._features = torch.cat(features).numpy()
        return self._features
    
    def add_suspect(self, name, labels):
        """
        添加一个"嫌疑人"标签序列。
        
        name: 嫌疑人名称, 如 'color', 'texture_source'
        labels: numpy array 或 tensor, 与 dataloader 同序
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self._suspect_labels[name] = np.asarray(labels)
    
    def test_suspect(self, name, n_trials=5):
        """
        对指定嫌疑人进行探针测试, 返回多次试验的平均准确率。
        
        训练线性分类器从特征预测嫌疑人标签。
        如果准确率显著高于随机基线, 说明模型内部确实编码了该特征。
        """
        if self._features is None:
            raise RuntimeError("请先调用 extract_features()")
        if name not in self._suspect_labels:
            raise KeyError(f"未知嫌疑人: {name}")
        
        X = self._features
        y = self._suspect_labels[name]
        n_classes = len(set(y))
        n_samples = len(X)
        n_train = int(n_samples * 0.8)
        baseline = 1.0 / n_classes
        
        accs = []
        for _ in range(n_trials):
            idx = np.random.permutation(n_samples)
            X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
            X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]
            
            probe = LogisticRegression(max_iter=500)
            probe.fit(X_train, y_train)
            accs.append(probe.score(X_test, y_test))
        
        return np.mean(accs), np.std(accs), baseline, n_classes
    
    def test_all_suspects(self, suspect_n_classes=None, n_trials=5):
        """测试所有嫌疑人, 返回 {name: {mean_acc, std, baseline, ...}}"""
        results = {}
        for name in self._suspect_labels:
            mean_acc, std, baseline, n_classes = self.test_suspect(name, n_trials)
            results[name] = {
                'probe_accuracy': mean_acc,
                'std': std,
                'baseline': baseline,
                'n_classes': n_classes,
                'ratio': mean_acc / baseline,
                'is_encoded': mean_acc > baseline * 1.5
            }
        self._probe_results = results
        return results
    
    def report(self):
        """生成探针分析报告"""
        if not self._probe_results:
            return "未运行探针分析。"
        
        lines = ["探针分析报告", "=" * 50]
        for name, r in self._probe_results.items():
            status = "已编码" if r['is_encoded'] else "未编码"
            lines.append(
                f"  {name}: 探针{r['probe_accuracy']:.3f} "
                f"(基线{r['baseline']:.3f}, 比率{r['ratio']:.1f}x) [{status}]"
            )
        return "\n".join(lines)
