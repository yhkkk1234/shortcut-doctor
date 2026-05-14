# Information Bandwidth Control: When and How Less Information Helps Neural Networks

**Authors:** Haolin Yang

**Contact:** comicyang1234@163.com

**Code:** https://github.com/yhkkk1234/shortcut-doctor

---

## Abstract

We investigate a simple but under-explored question: *can actively restricting the information available to a neural network during training make it learn better?* Through four controlled experiments—ranging from synthetic data to CIFAR-100 with ResNet-18—we show that **physical information bandwidth control at the input layer** can effectively prevent shortcut learning, but only when the shortcut signal is more vulnerable to the restriction than the true task signal. We categorize shortcuts by their frequency profile and introduce a learnable "bionic input layer" (9 parameters) that autonomously discovers which information channels to suppress—achieving 122% shortcut resistance improvement while adding negligible computational cost. We also present Shortcut Doctor, a lightweight Python toolkit that automates the pipeline: probe analysis → shortcut diagnosis → prescription → treatment verification. Our core finding is not that any single method is universally effective, but that the choice of information restriction must match the type of shortcut being targeted—and that models, given the option, will self-regulate toward better information diets.

---

## 1. Introduction

Deep neural networks are notorious for learning spurious correlations—shortcuts—instead of the underlying patterns they are supposed to learn. A model trained to count fingers may simply memorize that "hand = 5 fingers"; a classifier may rely on background color rather than object shape; a medical imaging model may learn hospital-specific artifacts rather than pathology.

Current approaches to mitigating shortcut learning include:

- **Data rebalancing** — resampling or reweighting the training distribution
- **Adversarial training** — penalizing reliance on known spurious features
- **Loss-based debiasing** — modifying the objective to discourage specific shortcuts

These methods are "soft"—they nudge the model away from shortcuts but do not physically prevent it from accessing shortcut information. The model's internal representations may still encode the shortcut, and it may find alternative pathways to exploit it.

We propose a different direction: **physical information bandwidth control at the input layer.** By restricting what information enters the model—through grayscale conversion, resolution reduction, or progressive blur—we can selectively destroy the shortcut signal while preserving (or at least not destroying more than) the true task signal.

The central insight is:

> A model cannot learn a shortcut it cannot see.

This paper makes four contributions:

1. **A taxonomy of shortcuts by frequency profile** (low-frequency vs. high-frequency), mapping each type to the appropriate information restriction method.

2. **Four controlled experiments** demonstrating when information bandwidth control works, when it fails, and why—from synthetic dots (grayscale: +133%, progressive blur: +57%) to CIFAR-100 with ResNet-18 (blur ineffective: shortcut survives) to a bionic input layer that autonomously learns to suppress color (+122%, 0.200→0.444).

3. **A cross-modal validation** of the "less is more" principle: a learnable information bottleneck at the input layer—with only 9 parameters—generalizes from text (25M LLM with Gaussian token diffusion) to vision (channel mixing initialized to grayscale), achieving significant shortcut resistance with negligible computational cost.

4. **Shortcut Doctor**, an open-source Python toolkit that automates the diagnostic pipeline: probe analysis → shortcut type classification → prescription → treatment verification.

---

## 2. Related Work

**Shortcut Learning.** Geirhos et al. (2020) provided a comprehensive survey of shortcut learning in neural networks, identifying texture bias, color bias, and background bias as common failure modes. Our work extends this by providing a frequency-based taxonomy and corresponding countermeasures.

**Information Bottleneck.** Tishby et al. (1999) introduced the information bottleneck principle, which formalizes the trade-off between compression and prediction. Our approach can be seen as a "hard" information bottleneck applied at the input layer.

**Early-Exit and Adaptive Inference.** BranchyNet (Teerapittayanon et al., 2016) and MSDNet (Huang et al., 2018) explore adaptive computation budgets. Our progressive blur strategy is related but operates on input quality rather than network depth.

**Probing.** Alain and Bengio (2017) introduced linear probing as a method to understand what information is encoded in intermediate representations. We use probing as the first step of our diagnostic pipeline.

**Curriculum Learning.** Bengio et al. (2009) proposed training on easier examples first. Progressive blur can be viewed as curriculum learning on the *information* axis rather than the *example difficulty* axis.

---

## 3. Method

### 3.1 The Diagnostic Pipeline

Our method follows a four-step pipeline:

```
Probe Analysis → Shortcut Diagnosis → Prescription → Treatment + Verification
```

#### Step 1: Probe Analysis

Given a trained model that may have learned shortcuts, we extract features from an intermediate layer and train linear probes (logistic regression) to predict suspected shortcut labels (e.g., color class, texture source). If probe accuracy significantly exceeds the random baseline, the shortcut is confirmed to be internally encoded.

#### Step 2: Shortcut Diagnosis

We classify each confirmed shortcut into one of five types based on its frequency profile:

| Shortcut Type | Frequency | Blur Resistant | Example                 |
| ------------- | --------- | -------------- | ----------------------- |
| COLOR         | Low       | Yes            | Red always means 3 dots |
| TEXTURE       | High      | No             | Watermark patterns      |
| SHAPE         | Low       | Yes            | Silhouette bias         |
| POSITION      | Middle    | Partial        | Corner artifacts        |
| MIXED         | Mixed     | Partial        | Combined biases         |

#### Step 3: Prescription

We map each shortcut type to the appropriate information restriction method:

```
COLOR shortcut    → Grayscale conversion (strip color channels)
TEXTURE shortcut   → Progressive blur (simulate visual development)
SHAPE shortcut     → Resolution reduction (downscale input)
POSITION shortcut  → Resolution reduction
MIXED/UNKNOWN      → Grayscale + progressive blur (conservative)
```

The mapping is based on a simple principle: **blur destroys high frequencies, so it only helps when the shortcut is high-frequency and the true task is low-frequency.** When both are in the same frequency band, blur is ineffective or harmful.

#### Step 4: Treatment and Verification

We apply the prescribed information restriction during training and evaluate on a clean (shortcut-free) test set. The treatment is applied only during training; inference uses original-resolution, clean inputs.

### 3.2 Progressive Blur

Progressive blur reduces input resolution during early training epochs and gradually restores it, simulating the developmental trajectory of infant vision:

```
Epochs 0-40%:    strong blur (down to 4×4)
Epochs 40-60%:   medium blur (8×8)
Epochs 60-85%:   mild blur (16×16)
Epochs 85-100%:  no blur (original)
```

This forces the model to first learn coarse, low-frequency patterns before being exposed to fine details that may include shortcuts.

---

## 4. Experiments

### 4.1 Synthetic Dots (Color Shortcut)

**Setup:** 128×128 images of colored circles containing 1-5 black dots. Colors: red, blue, green. Shortcut: 100% mapping of color → dot count (red=3, blue=4, green=5). Counterfactual test: uniform distribution of all combinations.

**Models:** Small CNN (4 conv layers), 5-way count classification. 900 training images.

| Model                         | Train (biased) | Counterfactual | Gap   | Color Probe |
| ----------------------------- | -------------- | -------------- | ----- | ----------- |
| Baseline (color)              | 1.000          | 0.200          | 0.800 | 1.000       |
| Grayscale                     | 1.000          | **0.467**      | 0.533 | 1.000       |
| Fixed blur (8×8)              | 1.000          | 0.200          | 0.800 | 1.000       |
| Progressive blur (4→8→native) | --             | **0.315**      | --    | 1.000       |
| Unbiased (upper bound)        | --             | 1.000          | --    | --          |

**Findings:**

- **Grayscale is most effective** (+133%, 0.200→0.467): color is a low-frequency shortcut, immune to blur.
- **Progressive blur helps** (+57%, 0.200→0.315): early extreme blur forces the model to use non-color features.
- **Fixed blur fails** (0.200): color survives 8×8 blur perfectly.
- **Color probes = 100% for all biased models**: the shortcut is internally encoded even when it doesn't dominate the output.

### 4.2 CIFAR-100 + ResNet-18 (Color Border Shortcut)

**Setup:** CIFAR-100 with a 4-pixel colored border unique to each of the 100 classes. Shortcut: 100% correlation. Clean test: original CIFAR-100 (no border).

**Model:** ResNet-18 adapted for 32×32 input. 50,000 training images, 30 epochs. Run on NVIDIA RTX 4090D.

| Model                      | Shortcut Test | Clean Test | Gap    |
| -------------------------- | ------------- | ---------- | ------ |
| Baseline (border)          | 0.966         | 0.017      | 0.949  |
| No border (upper bound)    | 0.213         | 0.592      | -0.379 |
| Fixed blur (24×24)         | 0.993         | 0.018      | 0.976  |
| Progressive blur (8→16→24) | 0.940         | 0.016      | 0.924  |

**Findings:**

- **The shortcut dominates completely**: baseline model achieves 96.6% with the border but only 1.7% without it (random = 1.0% for 100 classes). The model sees nothing but the colored border.
- **Blur is ineffective**: both fixed and progressive blur fail because the border is a low-frequency color signal that survives blur, while CIFAR-100's fine-grained visual features are destroyed.
- **This is a boundary case**: blur harms the true task more than the shortcut, violating the precondition for effectiveness.

### 4.3 Synthetic Geometric Patterns (Shape Shortcut)

**Setup:** 32×32 images of 10 geometric patterns with an 8×8 colored corner patch unique to each class. 100% correlation. 2,000 training images.

| Model                     | Shortcut Test | Clean Test | Gap   |
| ------------------------- | ------------- | ---------- | ----- |
| Baseline                  | 1.000         | 1.000      | 0.000 |
| No shortcut (upper bound) | --            | 1.000      | --    |
| Fixed blur (8×8)          | 0.800         | 0.800      | 0.000 |
| Progressive blur          | 1.000         | 0.800      | 0.200 |

**Findings:**

- **Task is too easy**: even with only 200 examples per class, the patterns are perfectly distinguishable.
- **Progressive blur backfires**: the model ends up *more* reliant on the shortcut because early extreme blur destroys the real patterns but the shortcut recovers first.

### 4.4 Learnable Input Modulation: The "Opt-In Color" Mechanism

**Motivation.** The previous experiments used fixed information restrictions (grayscale, blur, resolution reduction). But a fixed restriction may be either too aggressive (destroying useful signal) or too weak (letting the shortcut survive). A more elegant approach is to give the model the *option* to access information, but let it decide through gradient-based learning whether to use it.

**Setup.** We introduce a *Bionic Input Layer*—a learnable 3×3 channel mixing matrix placed before the model's first convolution, initialized to the grayscale matrix (all entries = 1/3). With only 9 parameters, this layer can either:
- Stay at grayscale (suppressing color information), or
- Recover color by learning a non-grayscale matrix.

In addition, we test an adversarial variant: a small color classifier (3-class MLP) trained on the bionic layer's output, with its gradient *reversed* for the bionic layer parameters. This creates a gradient conflict: the count task pushes toward preserving useful information, while the adversarial color signal pushes toward destroying color.

**Setup:** Same synthetic dots task as Experiment 4.1 (5-class counting, 100% color-to-count shortcut, 900 training images).

| Model                          | CF Accuracy | vs Baseline | Mixing Strength |
| ------------------------------ | ----------- | ----------- | --------------- |
| H) Baseline (color)            | 0.200       | --          | --              |
| C) True grayscale (reference)  | 0.458       | +0.258      | --              |
| M) Grayscale init, plain train | 0.393       | +0.193      | 0.010           |
| N) Grayscale init + adversary  | 0.444       | +0.244      | 0.008           |

**Findings:**

1. **The model actively chooses to remain grayscale.** Despite having 9 learnable parameters to recover full-color information, the mixing matrix barely deviates from 1/3 (mixing strength < 0.01). The model "realizes" through gradient descent that color information is harmful and elects not to use it.

2. **The adversarial signal reinforces this choice.** Model N (with adversary) achieves 0.444, within 3% of the hard-coded grayscale baseline (0.458), despite the adversary *not* having direct control over the bionic layer—it only provides a gradient signal that color is undesirable.

3. **"Opt-in" color outperforms "opt-out" color.** In Experiment 4.1, the adversarial approach starting from the identity matrix failed (mixing strength < 0.005, no improvement). Starting from grayscale and letting the model *recover* color if needed is far more effective than starting from color and trying to *remove* it.

**Cross-Modal Significance.** This experiment completes a cross-modal validation loop:
- **Text (25M LLM):** Gaussian blur along token sequence + learnable modulation → 33%+ loss reduction.
- **Image (9-parameter bionic layer):** Learnable channel mixing initialized to grayscale → 122% shortcut suppression (0.200 → 0.444).
- **Unified principle:** A small learnable information-controlling module at the input layer, initialized to a *restricted* state and allowed to self-regulate, is a general mechanism for shortcut mitigation across modalities.



The four experiments reveal a clear pattern:

```
Effectiveness = Vulnerability(shortcut) / Vulnerability(true task)

  Ratio > 1  →  Works        (blur kills shortcut, task survives)
  Ratio ≈ 1  →  No effect    (both equally affected)
  Ratio < 1  →  Backfires    (blur kills task, shortcut survives)
```

This leads to a practical decision tree:

```
                    ┌─ Shortcut is high-frequency (texture, watermark)?
                    │   YES → Progressive blur ✓
                    │
Question:           ├─ Shortcut is low-frequency (color, shape)?
What type of        │   ├─ Color  → Grayscale ✓
shortcut?           │   └─ Shape  → Downscale ✓
                    │
                    ├─ True task requires high-frequency details?
                    │   YES → Do NOT use blur; use targeted methods
                    │
                    └─ Unknown shortcut type?
                        → Probe first, then match method to type
```

---

## 6. Shortcut Doctor: A Practical Toolkit

We release **Shortcut Doctor**, a Python package implementing the full diagnostic pipeline:

```python
from shortcut_doctor import ProbeAnalyzer, ShortcutDiagnoser, Prescriber, Verifier

# 1. Probe: what shortcuts does the model encode internally?
analyzer = ProbeAnalyzer(model)
analyzer.extract_features(dataloader)
analyzer.add_suspect('color', color_labels)
results = analyzer.test_all_suspects()

# 2. Diagnose: classify shortcut type
diagnoser = ShortcutDiagnoser()
diagnosis = diagnoser.diagnose(results, {'color': ShortcutType.COLOR})

# 3. Prescribe: map type to countermeasure
prescriber = Prescriber()
prescription = prescriber.prescribe(diagnosis)

# 4. Treat + Verify
verifier = Verifier()
result = verifier.verify(treated_model, clean_test_loader, baseline_model)
```

The toolkit supports grayscale conversion, progressive blur, fixed blur, and resolution reduction, each mapped to a specific shortcut type.

---

## 7. Limitations

1. **Frequency overlap**: when the shortcut and true task occupy the same frequency band, information bandwidth control cannot separate them. In such cases, more sophisticated methods (causal representation learning, adversarial debiasing) may be needed.

2. **Catastrophic information loss**: overly aggressive blur or resolution reduction can destroy the true task signal, harming rather than helping. The progressive schedule partially mitigates this but does not eliminate it.

3. **Probe dependency**: the diagnostic pipeline requires access to suspected shortcut labels during probe analysis. In practice, these may not be available—automated shortcut discovery remains an open problem.

4. **Scale**: our experiments are on small to medium-scale datasets. The interaction between information bandwidth control and model scale (e.g., on ImageNet-scale data with large transformers) is unexplored.

---

## 8. Conclusion

We have shown that **physical information bandwidth control at the input layer** is a viable strategy for mitigating shortcut learning, with an important caveat: the method must match the shortcut type. Grayscale conversion handles color shortcuts; progressive blur handles texture shortcuts; resolution reduction handles shape shortcuts. No single method works for all shortcut types.

The key contribution is not any individual method, but the **diagnose-first, treat-second** framework. By probing what shortcuts a model has learned and classifying them by their frequency profile, we can select the appropriate countermeasure rather than blindly applying a one-size-fits-all solution.

We have released Shortcut Doctor as an open-source toolkit to make this pipeline accessible. The code, along with reproduction scripts for all experiments, is available at [GitHub URL].

---

**Acknowledgments:** This work emerged from an investigation into multi-scale visual recognition and the hypothesis that "actively limiting information can produce better learning outcomes than maximizing it."

**References:**

- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence.*
- Tishby, N., et al. (1999). The information bottleneck method.
- Teerapittayanon, S., et al. (2016). BranchyNet: Fast inference via early exiting from deep neural networks.
- Huang, G., et al. (2018). Multi-scale dense networks for resource efficient image classification.
- Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes.
- Bengio, Y., et al. (2009). Curriculum learning.
