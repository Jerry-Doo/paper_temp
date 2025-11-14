# PhaseMambaNet
### Physics-Aware Indirect Time-of-Flight (iToF) Network

**PhaseMambaNet** is a physics-guided neural network for indirect Time-of-Flight imaging.  
In *From-Depth* training mode, the model starts from ground-truth depth and RGB, synthesizes physically plausible 4-phase correlation measurements, and trains a U-shape backbone (with optional Mamba blocks) to jointly recover depth and a global correlation waveform.

---

## Key Features

### Differentiable Correlation Synthesizer
- 15 modulation frequencies (50 MHz × {1..19}, excluding multiples of 4).
- Learnable global waveform coefficients x_k = [a_1..a_K, s_1..s_K].
- Includes range falloff (1/r²), cosine-shading (cos^η θ), and per-pixel albedo.
- Converts depth to round-trip time using camera intrinsics.
- Environment-free formulation (no ambient light).

### Realistic Noise Model (S2 Path)
- Readout noise
- Shot noise
- Row/column fixed-pattern noise
- Gain/offset jitter
- N-bit quantization (default: 12-bit)
- Noise added before normalization to [0,1].

### Global Waveform Parameterization
- 30-D global learnable vector constrained by a post-box.
- Ensures waveform stability and hardware-feasible amplitude.

### RGB-Driven Albedo Estimation
- Lightweight CNN predicts per-pixel reflectance α(x,y).
- Supports sRGB-to-linear conversion.
- Helps separate geometry (depth) from appearance (albedo).

### U-Shape Backbone with Optional Mamba Blocks
Input:
- 4-phase (B,4,H,W), or
- 7-channel PhaseMix (DC, Re, Im)

Output:
- Depth map (B,1,H,W)
- Global waveform coefficients (B,30)

---

## Training Pipeline (From-Depth)
1. Convert ground-truth depth → round-trip time via intrinsics.
2. Synthesize clean 4-phase measurements.
3. Add noise + quantize + normalize.
4. Apply PhaseMix and feed into backbone.
5. Predict depth.
6. Re-synthesize 4-phase from predicted depth.
7. Compute physics-aware losses.

---

## Loss Functions
- Depth MAE  
- Depth SSIM  
- Multi-scale ZNCC  
- Frequency-weighted L1 prior on waveform parameters  

---
