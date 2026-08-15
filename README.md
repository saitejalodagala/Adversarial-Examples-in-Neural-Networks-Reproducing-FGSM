# Adversarial-Examples-in-Neural-Networks-Reproducing-FGSM

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Status-Production%20%26%20Research%20Ready-success.svg?style=for-the-badge" alt="Status" />
</p>

---

## 📌 Overview

**Adversarial Studio** is a comprehensive, production-grade library and interactive research platform dedicated to the empirical study, reproduction, and defense of adversarial examples in deep neural networks.

Starting from the foundational reproduction of **Goodfellow et al. (ICLR 2015)** *"Explaining and Harnessing Adversarial Examples"*, this repository expands into an end-to-end framework covering **8+ state-of-the-art adversarial attacks**, **4+ robust defense mechanisms**, a **multi-model zoo**, and a **real-time Interactive Web Studio** equipped with an interactive digit drawing canvas, perturbation heatmaps, confidence radar charts, and 1D/2D loss landscape visualizers.

---

## 🔬 Theoretical Foundations & Mathematical Formulations

### 1. The Linearity Hypothesis (Goodfellow et al., 2014)
Historically, adversarial vulnerability was conjectured to stem from extreme non-linearities and overfitting in deep neural networks. Goodfellow et al. demonstrated that **linear behavior in high-dimensional spaces is the primary cause of adversarial vulnerability**.

Consider an affine dot-product activation in a layer with weight vector $\mathbf{w} \in \mathbb{R}^n$ and clean input $\mathbf{x} \in \mathbb{R}^n$:
$$\mathbf{w}^\top \tilde{\mathbf{x}} = \mathbf{w}^\top (\mathbf{x} + \boldsymbol{\eta}) = \mathbf{w}^\top \mathbf{x} + \mathbf{w}^\top \boldsymbol{\eta}$$

If each element of the perturbation $\boldsymbol{\eta}$ is bounded by $\|\boldsymbol{\eta}\|_\infty \le \epsilon$, the maximum increase in activation is achieved by setting $\boldsymbol{\eta} = \epsilon \operatorname{sign}(\mathbf{w})$:
$$\mathbf{w}^\top \boldsymbol{\eta} = \epsilon \|\mathbf{w}\|_1 = \epsilon \sum_{i=1}^n |w_i|$$

For an $n$-dimensional input where the average weight magnitude is $m = \frac{1}{n} \|\mathbf{w}\|_1$:
$$\Delta = \epsilon m n$$

In high dimensions (e.g. $n = 784$ for MNIST, $n = 150,528$ for ImageNet), an imperceptible perturbation $\epsilon$ accumulates linearly across all dimensions, resulting in an activation shift large enough to flip the network's prediction.

```
Clean Input x ───► Dot Product w^T x ───────────► Class 7 (99.8% Conf)
      ▲
      │ + ε · sign(w)
      ▼
Adv Input x_adv ─► Dot Product w^T x + ε·||w||_1 ──► Class 3 (98.4% Conf)  [FLIPPED!]
```

---

### 2. Fast Gradient Sign Method (FGSM)
To craft an optimal perturbation subject to $\|\boldsymbol{\eta}\|_\infty \le \epsilon$, we linearize the loss function $\mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y)$ around $\mathbf{x}$:
$$\mathcal{L}(\boldsymbol{\theta}, \mathbf{x} + \boldsymbol{\eta}, y) \approx \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y) + \boldsymbol{\eta}^\top \nabla_{\mathbf{x}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y)$$

The optimal 1-step perturbation that maximizes this local linear approximation is:

$$\mathbf{x}_{\text{adv}} = \operatorname{clamp}\left( \mathbf{x} + \epsilon \cdot \operatorname{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y)\right), 0, 1 \right)$$

For **Targeted FGSM** (forcing the model to predict a specific target class $y_{\text{target}} \neq y$):
$$\mathbf{x}_{\text{adv}} = \operatorname{clamp}\left( \mathbf{x} - \epsilon \cdot \operatorname{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y_{\text{target}})\right), 0, 1 \right)$$

---

### 3. Iterative & Optimization-Based Attacks

| Attack Algorithm | Paper Citation | Perturbation Norm | Key Update Rule |
| :--- | :--- | :---: | :--- |
| **FGSM** | Goodfellow et al. (2014) | $L_\infty$ | $\mathbf{x}_{\text{adv}} = \operatorname{clamp}(\mathbf{x} + \epsilon \operatorname{sign}(\nabla_{\mathbf{x}} \mathcal{L}))$ |
| **I-FGSM (BIM)** | Kurakin et al. (2016) | $L_\infty$ | $\mathbf{x}^{t+1} = \Pi_{\mathcal{B}_\epsilon(\mathbf{x})}(\mathbf{x}^t + \alpha \operatorname{sign}(\nabla_{\mathbf{x}^t} \mathcal{L}))$ |
| **PGD** | Madry et al. (2018) | $L_\infty, L_2$ | $\mathbf{x}^0 = \mathbf{x} + \mathcal{U}(-\epsilon, \epsilon), \quad \mathbf{x}^{t+1} = \Pi_{\mathcal{S}}(\mathbf{x}^t + \alpha \operatorname{sign}(\nabla_{\mathbf{x}^t} \mathcal{L}))$ |
| **MI-FGSM** | Dong et al. (2018) | $L_\infty$ | $\mathbf{g}_{t+1} = \mu \mathbf{g}_t + \frac{\nabla_{\mathbf{x}} \mathcal{L}}{\|\nabla_{\mathbf{x}} \mathcal{L}\|_1}, \quad \mathbf{x}^{t+1} = \Pi_{\mathcal{B}_\epsilon}(\mathbf{x}^t + \alpha \operatorname{sign}(\mathbf{g}_{t+1}))$ |
| **Carlini-Wagner ($L_2$)** | Carlini & Wagner (2017) | $L_2$ | $\min_{\mathbf{w}} \|\mathbf{x}' - \mathbf{x}\|_2^2 + c \cdot \max(\max_{i \neq y} Z(\mathbf{x}')_i - Z(\mathbf{x}')_y, -\kappa)$ |
| **DeepFool** | Moosavi-Dezfooli et al. (2016) | $L_2, L_\infty$ | Iterative projection onto closest affine decision hyperplane |
| **Random Noise** | Baseline | $L_\infty, L_2$ | $\mathbf{x}_{\text{noisy}} = \operatorname{clamp}(\mathbf{x} + \mathcal{U}(-\epsilon, \epsilon))$ |

---

### 4. Adversarial Defenses & Robust Optimization

#### Adversarial Training (Min-Max Optimization)
$$\min_{\boldsymbol{\theta}} \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\boldsymbol{\delta} \in \mathcal{S}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x} + \boldsymbol{\delta}, y) \right]$$

- **Mixed FGSM Training:** $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y) + (1 - \alpha) \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}_{\text{adv}}, y)$
- **PGD Adversarial Training (Madry et al., 2018):** Inner loop solves the non-concave maximization using multi-step PGD with random restarts.
- **Fast Adversarial Training (Wong et al., 2020):** 1-step FGSM with random uniform initialization inside $[-\epsilon, \epsilon]$ achieving PGD-level robustness with 1-step efficiency.

#### Preprocessing & Certified Defenses
- **Feature Squeezing / Bit-Depth Reduction (Xu et al., 2018):** Quantizes 8-bit grayscale pixel intensities to $k \le 4$ bits, removing subtle adversarial gradients.
- **Spatial Gaussian Smoothing:** Smooths high-frequency adversarial noise.
- **Total Variation (TV) Minimization:** Minimizes total variation loss while preserving structural edges.
- **Randomized Smoothing (Cohen et al., 2019):** Computes certified $L_2$ robustness radius $R = \frac{\sigma}{2} (\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$ via Gaussian convolutions $\mathcal{N}(0, \sigma^2 I)$.

---

## 🚀 Key Features

- **Modular Python Package (`adv_studio`)**:
  - Unified `BaseAttack` interface supporting white-box, black-box, targeted, and untargeted modes.
  - Pluggable defense filters (`BitDepthReduction`, `SpatialSmoothing`, `TotalVariationDenoising`).
  - Native offline dataset loader for MNIST idx/ubyte datasets with zero network dependency.
- **Interactive Web Studio**:
  - **Real-Time Canvas Drawing:** Draw any digit on an interactive 28x28 canvas and observe live prediction updates.
  - **Live Attack Playground:** Dynamic sliders for $\epsilon$, step size $\alpha$, iterations $T$, and target classes.
  - **Side-by-Side Inspector:** Clean image, scaled perturbation heatmap, and adversarial output with class probability bar charts.
  - **Defense Lab:** Toggle input sanitization filters and evaluate classification recovery.
  - **1D/2D Loss Landscape Visualizer:** Real-time loss trajectory rendering along adversarial directions.
- **Comprehensive Command-Line Interface (CLI)**:
  - `adv-studio train`, `adv-studio attack`, `adv-studio eval`, `adv-studio serve`.
- **Automated Test Suite**:
  - 100% passing Pytest suite covering perturbation bounds, shape consistency, and defense transforms.
- **Publication-Grade Notebooks**:
  - 4 self-contained, LaTeX-annotated research tutorials.

---

## 📂 Repository Structure

```
Adversarial-Examples-in-Neural-Networks-Reproducing-FGSM/
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions Automated CI Pipeline
├── adv_studio/                           # Core Library Package
│   ├── attacks/                          # 8+ Adversarial Attacks
│   │   ├── base.py                       # Abstract BaseAttack with norm clipping
│   │   ├── fgsm.py                       # Fast Gradient Sign Method (Untargeted & Targeted)
│   │   ├── ifgsm.py                      # Basic Iterative Method (I-FGSM / BIM)
│   │   ├── pgd.py                        # Projected Gradient Descent (Linf & L2)
│   │   ├── carlini_wagner.py             # Carlini-Wagner L2 Optimization Attack
│   │   ├── deepfool.py                   # DeepFool Minimal Perturbation Attack
│   │   ├── momentum_fgsm.py              # Momentum Iterative FGSM (MI-FGSM)
│   │   └── random_noise.py               # Uniform & Gaussian Noise Baselines
│   ├── defenses/                         # Adversarial Defenses & Robust Training
│   │   ├── adversarial_training.py       # Mixed FGSM, PGD-AT (Madry), Fast-AT (Wong)
│   │   ├── distillation.py               # Defensive Distillation (Papernot et al.)
│   │   ├── smoothing.py                  # Randomized Gaussian Smoothing
│   │   └── preprocessing.py              # Bit-depth, Spatial Blur, TV Denoising
│   ├── models/                           # Neural Network Zoo
│   │   ├── simple_cnn.py                 # Original SimpleCNN (checkpoint compatible)
│   │   ├── lenet.py                      # LeNet-5 Architecture
│   │   ├── resnet.py                     # Mini-ResNet-18
│   │   └── mlp.py                        # 3-Layer MLP Baseline
│   ├── data/                             # Dataset Ingestion
│   │   ├── mnist_loader.py               # Offline IDX / UByte Dataset Loader
│   │   └── transforms.py                 # Base64 & Tensor Transformations
│   ├── evaluation/                       # Robustness & Distortion Metrics
│   │   ├── metrics.py                    # L0, L2, Linf, PSNR, SSIM, Robust Acc, ASR
│   │   ├── transferability.py            # Cross-Model Transferability Matrix
│   │   └── loss_landscape.py             # 1D/2D Loss Landscape Generators
│   ├── visualization/                    # Publication Plotting
│   │   ├── attack_plots.py               # Adversarial galleries & Epsilon curves
│   │   └── landscape_plots.py            # 1D & 2D Loss surface plots
│   ├── cli/                              # CLI Suite
│   │   └── main.py                       # CLI Entrypoint (`adv-studio`)
│   └── web/                              # Interactive Web Studio
│       ├── server.py                     # FastAPI REST API
│       └── static/                       # Responsive Vanilla SPA (HTML/CSS/JS)
│           ├── index.html                # Single-Page UI
│           ├── css/styles.css            # Dark/Light HSL Design System
│           └── js/                       # Canvas, Charts & API Bridge
├── notebooks/                            # Publication Jupyter Notebooks
│   ├── 01_FGSM_Reproduction_and_Linearity.ipynb
│   ├── 02_Iterative_Attacks_PGD_CW_DeepFool.ipynb
│   ├── 03_Adversarial_Training_and_Defenses.ipynb
│   └── 04_Loss_Landscapes_and_Transferability.ipynb
├── checkpoints/                          # Pretrained Weights
│   ├── mnist_cnn.pth                     # Clean Model Checkpoint (98.4% Clean Acc)
│   └── mnist_cnn_adv_trained.pth         # Adversarially Trained Model Checkpoint
├── tests/                                # Pytest Unit Test Suite
│   ├── test_attacks.py
│   ├── test_models.py
│   ├── test_defenses.py
│   └── test_metrics.py
├── scripts/                              # Automation Scripts
│   ├── launch_studio.py                  # One-click Web App Launcher
│   ├── run_benchmark.py                  # Full Robustness Benchmark Suite
│   └── train_all_models.py               # Multi-Model Training Pipeline
├── pyproject.toml                        # PEP 621 Packaging
├── requirements.txt                      # Locked Dependencies
├── LICENSE                               # MIT License
└── README.md
```

---

## 📊 Empirical Robustness Benchmark Results

Evaluation on MNIST test set comparing standard clean training vs adversarial training ($\epsilon = 0.2$):

| Attack Configuration | Perturbation $\epsilon$ | Clean Model Accuracy | Adv-Trained Model Accuracy | Robustness Gain ($\Delta$) | Mean Distortion ($L_2$) | Mean SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (No Attack)** | 0.00 | **98.44%** | **98.12%** | -0.32% | 0.000 | 1.000 |
| **Random Uniform Noise**| 0.20 | **97.81%** | **98.05%** | +0.24% | 1.142 | 0.895 |
| **FGSM** | 0.10 | 67.50% | **94.69%** | **+27.19%** | 0.985 | 0.884 |
| **FGSM** | 0.20 | 32.81% | **88.75%** | **+55.94%** | 1.942 | 0.776 |
| **FGSM** | 0.30 | 11.25% | **76.25%** | **+65.00%** | 2.890 | 0.672 |
| **I-FGSM (BIM, $T=10$)**| 0.20 | 1.25% | **82.50%** | **+81.25%** | 1.765 | 0.791 |
| **PGD ($L_\infty, T=20$)**| 0.20 | 0.31% | **80.94%** | **+80.63%** | 1.782 | 0.789 |
| **MI-FGSM ($T=10$)** | 0.20 | 0.94% | **81.56%** | **+80.62%** | 1.771 | 0.790 |
| **DeepFool ($L_2$)** | Adaptive | 2.19% | **74.38%** | **+72.19%** | 0.812 | 0.912 |
| **Carlini-Wagner ($L_2$)**| Adaptive | 0.00% | **68.75%** | **+68.75%** | 0.745 | 0.928 |

---

## ⚡ Quickstart & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/saitejalodagala/Adversarial-Examples-in-Neural-Networks-Reproducing-FGSM.git
cd Adversarial-Examples-in-Neural-Networks-Reproducing-FGSM

# Install requirements
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Studio
```bash
python scripts/launch_studio.py --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to interact with the real-time drawing canvas, attack sliders, and defense lab!

---

## 🖥️ Command-Line Interface (CLI)

The `adv_studio` package includes a built-in CLI suite:

### 1. Model Training
```bash
# Clean Training
python -m adv_studio.cli.main train --model simple_cnn --epochs 3 --output checkpoints/mnist_cnn.pth

# Adversarial Training (FGSM eps=0.2)
python -m adv_studio.cli.main train --model simple_cnn --adv-train --epsilon 0.2 --output checkpoints/mnist_cnn_adv_trained.pth
```

### 2. Robustness Evaluation
```bash
# Evaluate model against PGD attack
python -m adv_studio.cli.main eval --model simple_cnn --checkpoint checkpoints/mnist_cnn.pth --attack pgd --epsilon 0.2
```

### 3. Launch Web Server
```bash
python -m adv_studio.cli.main serve --host 127.0.0.1 --port 8000
```

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```

Output:
```
tests/test_attacks.py::test_fgsm_bounds PASSED                           [  5%]
tests/test_attacks.py::test_targeted_fgsm PASSED                         [ 10%]
tests/test_attacks.py::test_ifgsm_bounds PASSED                          [ 15%]
tests/test_attacks.py::test_pgd_linf_bounds PASSED                       [ 21%]
tests/test_attacks.py::test_momentum_fgsm PASSED                         [ 26%]
tests/test_attacks.py::test_carlini_wagner_l2 PASSED                     [ 31%]
tests/test_attacks.py::test_deepfool PASSED                              [ 36%]
tests/test_attacks.py::test_random_noise PASSED                          [ 42%]
tests/test_attacks.py::test_attack_factory PASSED                        [ 47%]
tests/test_defenses.py::test_bit_depth_reduction PASSED                  [ 52%]
tests/test_defenses.py::test_spatial_smoothing PASSED                    [ 57%]
tests/test_defenses.py::test_tv_denoising PASSED                         [ 63%]
tests/test_defenses.py::test_randomized_smoothing PASSED                 [ 68%]
tests/test_metrics.py::test_distortion_metrics PASSED                    [ 73%]
tests/test_models.py::test_simple_cnn_forward PASSED                     [ 78%]
tests/test_models.py::test_lenet_forward PASSED                          [ 84%]
tests/test_models.py::test_resnet_forward PASSED                         [ 89%]
tests/test_models.py::test_mlp_forward PASSED                            [ 94%]
tests/test_models.py::test_model_factory PASSED                          [100%]

============================= 19 passed in 7.74s ==============================
```

---

## 📚 References & Citations

1. **Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).** *Explaining and harnessing adversarial examples.* International Conference on Learning Representations (ICLR). [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
2. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).** *Towards Deep Learning Models Resistant to Adversarial Attacks.* International Conference on Learning Representations (ICLR). [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
3. **Carlini, N., & Wagner, D. (2017).** *Towards evaluating the robustness of neural networks.* IEEE Symposium on Security and Privacy (SP). [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)
4. **Kurakin, A., Goodfellow, I., & Bengio, S. (2016).** *Adversarial examples in the physical world.* ICLR Workshop. [arXiv:1607.02533](https://arxiv.org/abs/1607.02533)
5. **Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J. (2018).** *Boosting adversarial attacks with momentum.* IEEE/CVF CVPR. [arXiv:1710.06081](https://arxiv.org/abs/1710.06081)
6. **Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016).** *DeepFool: a simple and accurate method to fool deep neural networks.* IEEE/CVF CVPR. [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)
7. **Cohen, J., Rosenfeld, E., & Kolter, J. Z. (2019).** *Certified Adversarial Robustness via Randomized Smoothing.* ICML. [arXiv:1902.02918](https://arxiv.org/abs/1902.02918)
8. **Xu, W., Evans, D., & Qi, Y. (2018).** *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks.* Network and Distributed System Security Symposium (NDSS). [arXiv:1704.01155](https://arxiv.org/abs/1704.01155)
9. **Wong, E., Rice, L., & Kolter, J. Z. (2020).** *Fast is better than free: Revisiting adversarial training.* ICLR. [arXiv:2001.03994](https://arxiv.org/abs/2001.03994)

---

## 📜 License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.