# Physics-aware iToF (4‑phase) — From‑Depth Training + Measured Inference

本仓库实现了基于 **物理可微相关器** 的 iToF 深度重建。训练阶段仅需 **GT 深度**：
GT → 合成 4 相位 (含环境 + 几何衰减 falloff + 噪声) → Backbone(Mamba) → 粗深度 d1 →
再合成 → **残差** → **CL‑Refiner** → 最终深度 d2 → 自一致 **ZNCC** 监督。  
推理阶段输入 **实测四相位图**，按训练同样的闭环路径得到 d2。

---

## 0. 环境

本仓库提供 `environment.yml`（你给的版本）：

```yaml
name: cvpr
channels:
  - defaults
  - conda-forge
  - pytorch
channel_alias: https://mirrors.aliyun.com/anaconda/
show_channel_urls: true
dependencies:
  - python=3.9
  - numpy
  - scipy
  - matplotlib
  - scikit-learn
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit=11.3
  - pip
  - pip:
      - opencv-python
      - tqdm
      - tensorboard
