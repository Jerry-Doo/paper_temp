# model.py
# ============================================================
# Physics-aware iToF network with differentiable correlation
# - Global shared waveform x_k (cos+sin; post_box bounds)
# - Differentiable 4-phase synthesis (+ optional environment)
# - From-Depth training path with built-in paper-like sensor noise
# - Loss: MAE + SSIM + ZNCC (+ optional frequency prior)
# ============================================================

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- optional mamba backbone ----------
try:
    from mamba_ssm import Mamba  # pip install mamba-ssm
except Exception:
    Mamba = None


# -------------------- utilities --------------------
_SPEED_OF_LIGHT = 299_792_458.0  # m/s


def build_freq_list(base_hz: float = 50e6) -> torch.Tensor:
    """Base * [1..19] without multiples of 4 -> 15 freqs."""
    mults = [m for m in range(1, 20) if m % 4 != 0]
    return torch.tensor([base_hz * m for m in mults], dtype=torch.float32)


def depth_to_phase_t(depth_m: torch.Tensor) -> torch.Tensor:
    """depth (B,1,H,W) meters -> round-trip time t (B,1,H,W) seconds, t = 2d/c."""
    return 2.0 * depth_m / _SPEED_OF_LIGHT


# ---- SSIM (single-channel) ----
def _gaussian_window(channels: int, size: int, sigma: float, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    return w.repeat(channels, 1, 1, 1)

def ssim(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    C = img1.size(1)
    device, dtype = img1.device, img1.dtype
    w = _gaussian_window(C, window_size, sigma, device, dtype)
    mu1 = F.conv2d(img1, w, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, w, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sig1_sq = F.conv2d(img1 * img1, w, padding=window_size // 2, groups=C) - mu1_sq
    sig2_sq = F.conv2d(img2 * img2, w, padding=window_size // 2, groups=C) - mu2_sq
    sig12   = F.conv2d(img1 * img2, w, padding=window_size // 2, groups=C) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    num = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return (num / (den + 1e-8)).clamp(0.0, 1.0).mean()


# -------------------- correlation synthesizer --------------------
class CorrelationSynthesizer(nn.Module):
    r"""
    y(t) = b + Σ_k [ a_k cos(ω_k t + φ) + s_k sin(ω_k t + φ) ]
    4-phase sampling: φ ∈ {0, π/2, π, 3π/2}
    """
    def __init__(self,
                 base_freq_hz: float = 50e6,
                 normalize_mode: str = "none",
                 target_mean: float = 1.0,
                 target_l2: float = 1.0,
                 dc_baseline: float = 1.0,
                 enforce_l1_to_dc: bool = False,
                 use_soft_clip: bool = True,
                 soft_clip_temp: float = 6.0):
        super().__init__()
        freqs = build_freq_list(base_freq_hz)
        omega = 2.0 * math.pi * freqs
        self.register_buffer("freqs_hz", freqs)
        self.register_buffer("omega", omega)
        self.register_buffer("phase_shifts", torch.tensor(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=torch.float32))

        self.normalize_mode = normalize_mode
        self.target_mean = float(target_mean)
        self.target_l2 = float(target_l2)
        self.dc_baseline = float(dc_baseline)
        self.enforce_l1_to_dc = bool(enforce_l1_to_dc)
        self.use_soft_clip = bool(use_soft_clip)
        self.soft_clip_temp = float(soft_clip_temp)

    # ----- helpers -----
    def _split_coeffs(self, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xk = xk.squeeze()
        if xk.ndim == 1:
            xk = xk.unsqueeze(0)
        B, D = xk.shape
        K = int(self.omega.numel())
        if D == K:
            a = xk
            s = torch.zeros_like(a)
        elif D == 2 * K:
            a, s = xk[:, :K], xk[:, K:]
        else:
            raise ValueError(f"bad xk dim={D}, expect K={K} or 2K={2*K}")
        return a, s

    def _merge_coeffs(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, s], dim=-1)

    def _normalize_xk_post(self, xk: torch.Tensor) -> torch.Tensor:
        if not self.enforce_l1_to_dc:
            return xk
        b = self.dc_baseline
        a, s = self._split_coeffs(xk)
        l1 = (a.abs() + s.abs()).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = b / l1
        a = a * scale
        s = s * scale
        return self._merge_coeffs(a, s)

    def _soft_clip(self, y: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        temp = self.soft_clip_temp
        return lo + (hi - lo) * torch.sigmoid(temp * (y - lo) / max(1e-6, (hi - lo)))

    def _postprocess(self, y: torch.Tensor, dim: int, for_zncc: bool = False) -> torch.Tensor:
        b = self.dc_baseline
        if self.use_soft_clip and for_zncc:
            y = self._soft_clip(y, 0.0, 2.0 * b)
        else:
            y = y.clamp(0.0, 2.0 * b)

        mode = self.normalize_mode
        if mode == "none":
            return y
        if mode == "fixmean":
            mu = y.mean(dim=dim, keepdim=True).clamp_min(1e-12)
            return y * (self.target_mean / mu)
        if mode == "fixl2":
            E = y.pow(2).sum(dim=dim, keepdim=True).clamp_min(1e-12)
            return y * (self.target_l2 / E).sqrt()
        raise ValueError(f"bad normalize_mode={mode}")

    def to_unit(self, y: torch.Tensor) -> torch.Tensor:
        """Map 0..2b -> 0..1."""
        return (y / (2.0 * self.dc_baseline)).clamp(0.0, 1.0)

    # ----- continuous sampling -----
    def sample_continuous(self, xk: torch.Tensor, t_values: torch.Tensor,
                          phase_shift: float = 0.0, for_zncc: bool = False) -> torch.Tensor:
        xk = xk.squeeze()
        if xk.ndim == 1:
            xk = xk.unsqueeze(0)
        B = xk.size(0)
        xk = self._normalize_xk_post(xk)
        a, s = self._split_coeffs(xk)  # [B,K], [B,K]

        t_values = t_values.to(xk.device).squeeze()
        if t_values.ndim == 1:
            t_values = t_values.unsqueeze(0).expand(B, -1)  # [B,T]
        elif t_values.size(0) != B:
            raise ValueError(f"t_values batch ({t_values.size(0)}) != xk batch ({B})")

        K = a.size(1)
        omega = self.omega[:K].to(xk.device).view(1, K, 1)
        t = t_values.view(B, 1, -1)
        ph = torch.as_tensor(phase_shift, dtype=torch.float32, device=xk.device).view(1, 1, 1)

        basis_c = torch.cos(omega * t + ph)  # [B,K,T]
        basis_s = torch.sin(omega * t + ph)  # [B,K,T]
        y = self.dc_baseline + (a.unsqueeze(-1) * basis_c + s.unsqueeze(-1) * basis_s).sum(dim=1)  # [B,T]
        return self._postprocess(y, dim=-1, for_zncc=for_zncc)

    # ----- 4-phase per-pixel sampling (signal-only) -----
    def sample_map(self, xk: torch.Tensor, t_map: torch.Tensor, for_zncc: bool = False) -> torch.Tensor:
        """
        xk:   [K] or [2K] or [B,K] or [B,2K]
        t_map:[H,W] or [B,H,W] or [B,1,H,W]  (seconds)
        return: (B,4,H,W), in [0, 2b] after postprocess
        """
        xk = xk.squeeze()
        if xk.ndim == 1:
            xk = xk.unsqueeze(0)  # [1,D]
        xk = self._normalize_xk_post(xk)
        a, s = self._split_coeffs(xk)  # [B,K],[B,K]

        # t_map -> [B,1,H,W]
        if t_map.ndim == 2:
            t_map = t_map.unsqueeze(0).unsqueeze(0)
        elif t_map.ndim == 3:
            t_map = t_map.unsqueeze(1)
        elif t_map.ndim != 4:
            raise ValueError(f"bad t_map shape {tuple(t_map.shape)}")

        device = t_map.device
        a, s = a.to(device), s.to(device)

        # ---- robust batch broadcasting ----
        Bx = a.size(0)
        Bt = t_map.size(0)
        if Bx == 1 and Bt > 1:
            a = a.expand(Bt, -1)
            s = s.expand(Bt, -1)
        elif Bx != Bt:
            raise ValueError(f"batch mismatch xk({Bx}) vs t_map({Bt})")

        B = Bt
        K = a.size(1)
        omega = self.omega[:K].to(device)               # (K,)
        phase = self.phase_shifts.to(device)            # (4,)

        t  = t_map.unsqueeze(1)                         # (B,1,1,H,W)
        w  = omega.view(1, K, 1, 1, 1)                  # (1,K,1,1,1)
        ph = phase.view(1, 1, 4, 1, 1)                  # (1,1,4,1,1)

        basis_c = torch.cos(w * t + ph)                 # (B,K,4,H,W)
        basis_s = torch.sin(w * t + ph)                 # (B,K,4,H,W)
        y = self.dc_baseline + (a.view(B, K, 1, 1, 1) * basis_c
                                + s.view(B, K, 1, 1, 1) * basis_s).sum(dim=1)  # (B,4,H,W)
        return self._postprocess(y, dim=1, for_zncc=for_zncc)

    # ----- 4-phase with environment -----
    def sample_map_env(self,
                       xk: torch.Tensor,
                       t_map: torch.Tensor,
                       beta: torch.Tensor | float = 0.0,          # ambient
                       kappa: torch.Tensor | None = None,          # per-phase DC (4,) or (B,4,1,1)
                       lam_d: torch.Tensor | float = 0.0,          # dark current
                       gain: torch.Tensor | float = 1.0,           # readout/exposure gain
                       alpha: torch.Tensor | float | None = None,  # reflectance (optional)
                       use_falloff: bool = False, d0: float = 1.0, # geometric falloff (optional)
                       for_zncc: bool = False) -> torch.Tensor:
        """
        out = gain * ( alpha*falloff*y_signal + beta*kappa + lam_d )
        """
        # signal (already postprocessed/clipped)
        y_signal = self.sample_map(xk, t_map, for_zncc=for_zncc)  # (B,4,H,W) in [0,2b]

        B, _, H, W = y_signal.shape
        device = y_signal.device
        dtype  = y_signal.dtype

        # alpha & falloff
        if alpha is None:
            alpha = 1.0
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=dtype, device=device)
        if alpha.ndim == 0:
            alpha = alpha.view(1,1,1,1).expand(B,1,H,W)

        if use_falloff:
            if t_map.ndim == 4:
                z_approx = t_map * (_SPEED_OF_LIGHT / 2.0)
            elif t_map.ndim == 3:
                z_approx = t_map.unsqueeze(1) * (_SPEED_OF_LIGHT / 2.0)
            else:
                z_approx = t_map.unsqueeze(0).unsqueeze(0) * (_SPEED_OF_LIGHT / 2.0)
            z_approx = z_approx.squeeze(1).clamp_min(1e-6)
            falloff = (d0 / z_approx).pow(2.0).clamp_max(1e6).view(B,1,H,W)
        else:
            falloff = 1.0

        if not torch.is_tensor(beta):
            beta = torch.tensor(beta, dtype=dtype, device=device)
        beta = beta.view(1,1,1,1)

        if kappa is None:
            kappa = torch.ones(4, dtype=dtype, device=device)
        if kappa.ndim == 1:               # (4,)
            kappa = kappa.view(1,4,1,1).expand(B,4,H,W)
        elif kappa.ndim == 4:
            if kappa.shape[-2:] == (1,1):
                kappa = kappa.expand(B,4,H,W)
        else:
            raise ValueError(f"bad kappa shape {tuple(kappa.shape)}")

        if not torch.is_tensor(lam_d):
            lam_d = torch.tensor(lam_d, dtype=dtype, device=device)
        lam_d = lam_d.view(1,1,1,1)

        if not torch.is_tensor(gain):
            gain = torch.tensor(gain, dtype=dtype, device=device)
        gain = gain.view(1,1,1,1)

        pre = (alpha * falloff) * y_signal + beta * kappa + lam_d   # (B,4,H,W)
        out = gain * pre
        return out


# -------------------- backbones --------------------
class MambaBlock2D(nn.Module):
    """Wrap mamba-ssm.Mamba for 2D features (flatten HW as sequence)."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm not found. Please: pip install mamba-ssm")
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B,L,C)
        y = self.norm(y)
        y = self.mamba(y) + y
        y = self.ffn(y) + y
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y


class ConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU()
        )
    def forward(self, x): return self.net(x)


class Backbone(nn.Module):
    """
    PatchEmbed -> N blocks -> Upsample -> heads
    If mamba-ssm is available, uses Mamba blocks; otherwise conv blocks.
    """
    def __init__(self, in_ch: int = 4, dim: int = 128, num_blocks: int = 8, scale: int = 4, k: int = 30):
        super().__init__()
        self.scale = scale
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=7, stride=scale, padding=3),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        if Mamba is not None:
            blocks = [MambaBlock2D(dim) for _ in range(num_blocks)]
        else:
            blocks = [ConvBlock(dim) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.xk_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(),
            nn.Conv2d(dim, k, 1)
        )

    def forward(self, x: torch.Tensor):
        feat_l = self.embed(x)
        feat_l = self.blocks(feat_l)
        feat = self.upsample(feat_l)
        depth_raw = self.depth_head(feat)                 # (B,1,H,W)
        xk_raw = self.xk_head(self.gap(feat)).flatten(1)  # (B,K) (not used when global)
        return feat, depth_raw, xk_raw


# -------------------- main network --------------------
class PhaseMambaNet(nn.Module):
    def __init__(self,
                 dim: int = 128,
                 num_blocks: int = 8,
                 in_ch: int = 4,
                 use_corr_synth: bool = True,
                 base_freq_hz: float = 50e6,
                 dmax: float = 8.0,
                 # physics layer
                 corr_norm_mode: str = "none",
                 corr_target_mean: float = 1.0,
                 corr_target_l2: float = 1.0,
                 corr_dc_baseline: float = 1.0,
                 corr_enforce_l1_to_dc: bool = False,
                 corr_use_soft_clip: bool = True,
                 corr_soft_clip_temp: float = 6.0,
                 # global waveform
                 use_global_xk: bool = True,
                 xk_param_mode: str = "post_box",     # "built_in_l1" | "post_box"
                 xk_b_total: float = 1.0,
                 xk_min: Optional[torch.Tensor] = None,
                 xk_max: Optional[torch.Tensor] = None):
        super().__init__()
        self.use_corr_synth = use_corr_synth
        self.dmax = float(dmax)
        self.use_global_xk = bool(use_global_xk)
        self.xk_param_mode = xk_param_mode
        self.xk_b_total = float(xk_b_total)

        # K freqs
        self.k_freq = int(build_freq_list(base_freq_hz).numel())
        self.k_param = 2 * self.k_freq  # cos+sin

        # physics
        self.corr = CorrelationSynthesizer(
            base_freq_hz=base_freq_hz,
            normalize_mode=corr_norm_mode,
            target_mean=corr_target_mean,
            target_l2=corr_target_l2,
            dc_baseline=corr_dc_baseline,
            enforce_l1_to_dc=corr_enforce_l1_to_dc,
            use_soft_clip=corr_use_soft_clip,
            soft_clip_temp=corr_soft_clip_temp,
        )

        # backbone
        self.backbone = Backbone(in_ch=in_ch, dim=dim, num_blocks=num_blocks, k=self.k_param)

        # waveform params (post_box mode expected)
        if xk_param_mode == "built_in_l1":
            self.p_logits = nn.Parameter(torch.zeros(self.k_param))
            self.q_logits = nn.Parameter(torch.zeros(self.k_param))
            nn.init.normal_(self.p_logits, std=0.02)
            nn.init.normal_(self.q_logits, std=0.02)
            self.register_buffer("xk_min", torch.zeros(1, self.k_param))
            self.register_buffer("xk_max", torch.zeros(1, self.k_param))
        elif xk_param_mode == "post_box":
            if xk_min is None or xk_max is None:
                raise ValueError("post_box mode requires xk_min/xk_max")
            xk_min = xk_min.flatten()
            xk_max = xk_max.flatten()
            if xk_min.numel() not in (self.k_freq, 2*self.k_freq):
                raise ValueError(f"xk_min/xk_max length must be K={self.k_freq} or 2K={2*self.k_freq}")
            if xk_min.numel() == self.k_freq:
                xk_min = torch.cat([xk_min, xk_min], dim=0)
                xk_max = torch.cat([xk_max, xk_max], dim=0)
            self.register_buffer("xk_min", xk_min.view(1, self.k_param))
            self.register_buffer("xk_max", xk_max.view(1, self.k_param))
            self.global_xk_logits = nn.Parameter(torch.zeros(self.k_param))
            with torch.no_grad():
                mid = 0.5 * (self.xk_min + self.xk_max)
                rng = (self.xk_max - self.xk_min).clamp_min(1e-6)
                init = (mid - self.xk_min) / rng
                self.global_xk_logits.copy_(torch.log(init / (1 - init)).squeeze())
        else:
            raise ValueError("xk_param_mode must be 'built_in_l1' or 'post_box'")

        # learnable environment (global)
        self.learn_env = True
        self.env_beta  = nn.Parameter(torch.tensor(0.0))
        self.env_lam_d = nn.Parameter(torch.tensor(0.0))
        self.env_gain  = nn.Parameter(torch.tensor(1.0))
        self.env_kappa = nn.Parameter(torch.ones(4))

        # built-in "paper-like" sensor noise (only used in from-depth train path)
        self._noise_read = (0.001, 0.01)
        self._noise_shot = (0.0,   0.02)
        self._noise_gain = (0.95,  1.05)
        self._noise_offs = (-0.02, 0.02)
        self._noise_fpnr = (0.0,   0.005)
        self._noise_fpnc = (0.0,   0.005)

    # ----- helper: paper-like noise -----
    def paper_noise(self, img4: torch.Tensor) -> torch.Tensor:
        """
        Add read/shot/gain/offset + row/col FPN + 12-bit quantization.
        img4: (B,4,H,W) in [0,1]
        """
        x = img4.clamp(0.0, 1.0)
        B, C, H, W = x.shape
        device = x.device

        def _rand(a,b): return float(torch.empty(1, device=device).uniform_(a,b))
        gain   = _rand(*self._noise_gain)
        offset = _rand(*self._noise_offs)
        y = x * gain + offset

        read_std = _rand(*self._noise_read)
        y = y + torch.randn_like(y) * read_std

        shot_k = _rand(*self._noise_shot)
        y = y + torch.randn_like(y) * torch.sqrt(torch.clamp(y, min=0.0)) * shot_k

        row_amp = _rand(*self._noise_fpnr)
        if row_amp > 0.0:
            y = y + torch.randn(B, 1, H, 1, device=device) * row_amp
        col_amp = _rand(*self._noise_fpnc)
        if col_amp > 0.0:
            y = y + torch.randn(B, 1, 1, W, device=device) * col_amp

        levels = (1 << 12) - 1
        y = torch.round(y * levels) / levels
        return y.clamp(0.0, 1.0)

    # ----- waveform parameterization -----
    def _xk_from_built_in_l1(self) -> torch.Tensor:
        amp = self.xk_b_total * torch.softmax(self.p_logits, dim=-1)   # >=0, sum=b
        sgn = torch.tanh(self.q_logits)                                # (-1,1)
        return (amp * sgn).unsqueeze(0)                                 # [1,2K]

    def _xk_from_post_box(self) -> torch.Tensor:
        xk_sig = torch.sigmoid(self.global_xk_logits).unsqueeze(0)      # [1,2K]
        rng = (self.xk_max - self.xk_min)
        return self.xk_min + rng * xk_sig

    def get_global_xk(self) -> torch.Tensor:
        if self.xk_param_mode == "built_in_l1":
            return self._xk_from_built_in_l1()
        else:
            return self._xk_from_post_box()

    # ----- measured mode -----
    def forward(self, four_phase: torch.Tensor):
        device = next(self.parameters()).device
        four_phase = four_phase.to(device)

        _, depth_raw, _ = self.backbone(four_phase)
        pred_depth = (torch.tanh(depth_raw) * 0.5 + 0.5) * self.dmax

        if self.use_global_xk:
            xk = self.get_global_xk().to(device)            # [1,2K]
            pred_xk = xk.expand(four_phase.size(0), -1)     # [B,2K]
        else:
            raise NotImplementedError("Per-sample x_k disabled; set use_global_xk=True.")

        extras: Dict[str, torch.Tensor] = {}
        if self.use_corr_synth:
            t_map = depth_to_phase_t(pred_depth)
            # signal-only 4-phase
            pred_four = self.corr.sample_map(pred_xk, t_map, for_zncc=False)
            extras["pred_four_phase"]   = pred_four
            extras["pred_four_phase01"] = self.corr.to_unit(pred_four)
            # env-aware (for zncc/loss)
            pred_four_env = self.corr.sample_map_env(
                pred_xk, t_map,
                beta=self.env_beta, kappa=self.env_kappa,
                lam_d=self.env_lam_d, gain=self.env_gain,
                alpha=None, use_falloff=False, for_zncc=True
            )
            extras["pred_four_phase_env"]   = pred_four_env
            extras["pred_four_phase_env01"] = self.corr.to_unit(pred_four_env)

        return pred_xk, pred_depth, extras

    # ----- from-depth training path (paper style) -----
    def forward_from_depth_train(self, gt_depth: torch.Tensor,
                                 to_unit: bool = True, use_falloff: bool = False):
        """
        1) use GT depth -> synth I_obs (env) -> add paper-like noise (train only)
        2) backbone(I_obs) -> pred_depth
        3) pred_depth -> synth pred_four_env (for zncc consistency)
        """
        device = next(self.parameters()).device
        gt_depth = gt_depth.to(device)

        # (1) synth observation from GT depth
        t_map_obs = depth_to_phase_t(gt_depth)
        xk = self.get_global_xk().to(device)          # [1,2K]
        B  = gt_depth.size(0)
        xkB = xk.expand(B, -1)                        # ★ expand to batch
        I_obs_env = self.corr.sample_map_env(
            xkB, t_map_obs,
            beta=self.env_beta, kappa=self.env_kappa,
            lam_d=self.env_lam_d, gain=self.env_gain,
            alpha=None, use_falloff=use_falloff, for_zncc=True
        )
        I_obs01 = self.corr.to_unit(I_obs_env) if to_unit else I_obs_env
        if self.training:
            I_obs01 = self.paper_noise(I_obs01)

        # (2) predict depth
        _, depth_raw, _ = self.backbone(I_obs01)
        pred_depth = (torch.tanh(depth_raw) * 0.5 + 0.5) * self.dmax

        # (3) synth from predicted depth for self-consistency
        t_map_pred = depth_to_phase_t(pred_depth)
        pred_four_env = self.corr.sample_map_env(
            xkB, t_map_pred,
            beta=self.env_beta, kappa=self.env_kappa,
            lam_d=self.env_lam_d, gain=self.env_gain,
            alpha=None, use_falloff=use_falloff, for_zncc=True
        )

        extras = {
            "I_obs_paper_like01": I_obs01,
            "pred_four_phase_env":  pred_four_env,
            "pred_four_phase_env01": self.corr.to_unit(pred_four_env),
        }
        return xkB, pred_depth, extras


# -------------------- loss --------------------
class PhaseMambaPhysLoss(nn.Module):
    def __init__(self,
                 mae_w: float = 1.0,
                 ssim_w: float = 0.5,
                 zncc_w: float = 0.5,
                 edge_w: float = 0.0,
                 # SSIM config
                 ssim_use_depth_unit: bool = True,  # if True, divide depth by dmax before SSIM
                 ssim_max_val: float = 1.0,
                 dmax_for_ssim: float = 6.0,
                 # physics for fallback synthesis
                 base_freq_hz: float = 50e6,
                 corr_norm_mode: str = "none",
                 corr_target_mean: float = 1.0,
                 corr_target_l2: float = 1.0,
                 corr_dc_baseline: float = 1.0,
                 corr_enforce_l1_to_dc: bool = False,
                 corr_use_soft_clip: bool = True,
                 corr_soft_clip_temp: float = 6.0,
                 # frequency prior
                 freq_l1_w: float = 0.0,
                 freq_alpha: float = 1.0,
                 # options
                 zncc_to_depth: bool = True):
        super().__init__()
        self.mae_w = mae_w
        self.ssim_w = ssim_w
        self.zncc_w = zncc_w
        self.edge_w = edge_w

        self.ssim_use_depth_unit = ssim_use_depth_unit
        self.ssim_max_val = ssim_max_val
        self.dmax_for_ssim = dmax_for_ssim

        self.zncc_to_depth = bool(zncc_to_depth)

        # physics (fallback only)
        self.corr = CorrelationSynthesizer(
            base_freq_hz=base_freq_hz,
            normalize_mode=corr_norm_mode,
            target_mean=corr_target_mean,
            target_l2=corr_target_l2,
            dc_baseline=corr_dc_baseline,
            enforce_l1_to_dc=corr_enforce_l1_to_dc,
            use_soft_clip=corr_use_soft_clip,
            soft_clip_temp=corr_soft_clip_temp,
        )
        self._dc = float(corr_dc_baseline)

        # frequency prior
        self.freq_l1_w = float(freq_l1_w)
        freqs = build_freq_list(base_freq_hz)
        w = (freqs.float() / freqs.max()).pow(freq_alpha)
        self.register_buffer("freq_weights", w)

    @staticmethod
    def zncc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Pixelwise ZNCC along phase dim (C=4). Expect a,b in 0..1."""
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        a_std = a.std(dim=1, keepdim=True) + eps
        b_std = b.std(dim=1, keepdim=True) + eps
        corr = ((a * b).sum(dim=1) / (a_std * b_std)).clamp(-1, 1)
        return corr.mean()

    def _freq_l1_prior(self, xk: torch.Tensor) -> torch.Tensor:
        if self.freq_l1_w <= 0:
            return xk.new_tensor(0.0)
        xk = xk.squeeze()
        if xk.ndim == 1:
            xk = xk.unsqueeze(0)
        B, D = xk.shape
        K = self.freq_weights.numel()
        assert D in (K, 2*K)
        if D == 2*K:
            a, s = xk[:, :K], xk[:, K:]
        else:
            a, s = xk, xk.new_zeros(xk.shape)
        w = self.freq_weights.view(1, K)
        return (w * (a.abs() + s.abs())).sum() / B

    def forward(self,
                pred_depth: torch.Tensor,                 # (B,1,H,W)
                gt_depth: torch.Tensor,                   # (B,1,H,W)
                pred_xk: Optional[torch.Tensor] = None,
                obs_four_phase: Optional[torch.Tensor] = None,       # (B,4,H,W) or 0..1
                pred_four_phase_pred01: Optional[torch.Tensor] = None # (B,4,H,W) 0..1
                ):
        device = pred_depth.device
        gt_depth = gt_depth.to(device)

        # MAE
        mae = F.l1_loss(pred_depth, gt_depth)

        # SSIM
        if self.ssim_use_depth_unit:
            pd = (pred_depth / max(1e-6, self.dmax_for_ssim)).clamp(0, 1)
            gd = (gt_depth   / max(1e-6, self.dmax_for_ssim)).clamp(0, 1)
            ssim_val = ssim(pd, gd, max_val=self.ssim_max_val).clamp(0.0, 1.0)
        else:
            ssim_val = ssim(pred_depth, gt_depth, max_val=self.dmax_for_ssim).clamp(0.0, 1.0)
        ssim_loss = 1.0 - ssim_val

        zncc_loss = torch.tensor(0.0, device=device)
        freq_prior = torch.tensor(0.0, device=device)

        # ZNCC path (depth->4phase self-consistency)
        if self.zncc_w > 0 and (pred_xk is not None) and (obs_four_phase is not None):
            obs = obs_four_phase.to(device)
            if pred_four_phase_pred01 is not None:
                pred_four = pred_four_phase_pred01.to(device)
            else:
                # fallback: synth from pred_depth
                t_map = depth_to_phase_t(pred_depth.detach())
                pred_four_sig = self.corr.sample_map(pred_xk, t_map, for_zncc=True)  # 0..2b
                pred_four = self.corr.to_unit(pred_four_sig)

            # ensure 0..1 for ZNCC
            if obs.max() > 1.0 + 1e-6 or obs.min() < -1e-6:
                obs = (obs / (2.0 * self._dc)).clamp(0, 1)

            zncc_val = self.zncc(pred_four, obs)
            zncc_loss = 1.0 - zncc_val

            # frequency prior
            if pred_xk is not None and self.freq_l1_w > 0:
                freq_prior = self._freq_l1_prior(pred_xk)

        total = (self.mae_w * mae
                 + self.ssim_w * ssim_loss
                 + self.zncc_w * zncc_loss
                 + self.freq_l1_w * freq_prior)

        logs = {
            "mae": float(mae.detach()),
            "ssim": float(ssim_val.detach()),
            "zncc": float((1.0 - zncc_loss).detach()) if zncc_loss.numel() else 0.0,
            "freq_prior": float(freq_prior.detach()) if freq_prior.numel() else 0.0,
            "total": float(total.detach()),
        }
        return total, logs
