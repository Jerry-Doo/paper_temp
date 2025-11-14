from __future__ import annotations
import math, contextlib
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_MAMBA = True
try:
    from mamba_ssm import Mamba 
except Exception:
    _HAS_MAMBA = False

_SPEED_OF_LIGHT = 299_792_458.0 


def build_freq_list(base_hz: float = 50e6) -> torch.Tensor:
    """Base * [1..19] without multiples of 4 -> 15 freqs."""
    mults = [m for m in range(1, 19 + 1) if m % 4 != 0]
    return torch.tensor([base_hz * m for m in mults], dtype=torch.float32)


def depth_to_phase_t(depth_m: torch.Tensor) -> torch.Tensor:
    """depth (B,1,H,W) meters -> round-trip time t (B,1,H,W) seconds, t = 2d/c."""
    return 2.0 * depth_m / _SPEED_OF_LIGHT


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def _nan_safe(x: torch.Tensor, minv: float | None = None, maxv: float | None = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if (minv is not None) or (maxv is not None):
        lo = -float("inf") if minv is None else float(minv)
        hi = float("inf") if maxv is None else float(maxv)
        x = x.clamp(lo, hi)
    return x

def _gaussian_window(channels: int, size: int, sigma: float, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0) 
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

def _parse_intrinsics(K: Optional[torch.Tensor] = None,
                      fx: Optional[float] = None, fy: Optional[float] = None,
                      cx: Optional[float] = None, cy: Optional[float] = None,
                      device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if K is not None:
        if K.ndim == 3:
            K = K[0]
        assert K.shape == (3, 3), f"Bad K shape {tuple(K.shape)}"
        fx_t = torch.as_tensor(K[0, 0], dtype=dtype, device=device)
        fy_t = torch.as_tensor(K[1, 1], dtype=dtype, device=device)
        cx_t = torch.as_tensor(K[0, 2], dtype=dtype, device=device)
        cy_t = torch.as_tensor(K[1, 2], dtype=dtype, device=device)
    else:
        if None in (fx, fy, cx, cy):
            raise ValueError("Must provide either K or all of (fx, fy, cx, cy).")
        fx_t = torch.as_tensor(fx, dtype=dtype, device=device)
        fy_t = torch.as_tensor(fy, dtype=dtype, device=device)
        cx_t = torch.as_tensor(cx, dtype=dtype, device=device)
        cy_t = torch.as_tensor(cy, dtype=dtype, device=device)
    return fx_t, fy_t, cx_t, cy_t


def _ray_norm_grid(H: int, W: int,
                   fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                   device, dtype) -> torch.Tensor:
    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    xd = (xs - cx) / fx
    yd = (ys - cy) / fy
    rn = torch.sqrt(1.0 + xd * xd + yd * yd)
    return rn.view(1, 1, H, W)


def depth_to_phase_t_intrinsics(depth_m: torch.Tensor,
                                K: Optional[torch.Tensor] = None,
                                fx: Optional[float] = None, fy: Optional[float] = None,
                                cx: Optional[float] = None, cy: Optional[float] = None,
                                depth_is_z: bool = True) -> torch.Tensor:
    depth_m = _nan_safe(depth_m)
    B, C, H, W = depth_m.shape
    assert C == 1, f"expect depth shape (B,1,H,W), got {tuple(depth_m.shape)}"
    device, dtype = depth_m.device, depth_m.dtype
    if depth_is_z:
        fx_t, fy_t, cx_t, cy_t = _parse_intrinsics(K, fx, fy, cx, cy, device=device, dtype=dtype)
        rn = _ray_norm_grid(H, W, fx_t, fy_t, cx_t, cy_t, device, dtype)  # (1,1,H,W)
        r = depth_m * rn
    else:
        r = depth_m
    return 2.0 * r / _SPEED_OF_LIGHT


def cosine_from_intrinsics(H: int, W: int,
                           intrinsics: Union[torch.Tensor, Tuple[float, float, float, float], dict],
                           device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(intrinsics, dict):
        fx, fy, cx, cy = float(intrinsics["fx"]), float(intrinsics["fy"]), float(intrinsics["cx"]), float(intrinsics["cy"])
    elif torch.is_tensor(intrinsics):
        K = intrinsics.to(dtype=torch.float32, device=device)
        if K.numel() == 9:
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        elif K.numel() == 4:
            fx, fy, cx, cy = [float(v) for v in K.view(-1)]
        else:
            raise ValueError("intrinsics tensor must be 3x3 or length-4 (fx,fy,cx,cy)")
    else:
        fx, fy, cx, cy = [float(v) for v in intrinsics]
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    UU, VV = torch.meshgrid(u, v, indexing='xy')
    x = (UU - cx) / max(1e-6, fx)
    y = (VV - cy) / max(1e-6, fy)
    cos = 1.0 / torch.sqrt(1.0 + x * x + y * y) 
    return cos.clamp(0.0, 1.0).view(1, 1, H, W)



def phasemix_cat(I: torch.Tensor) -> torch.Tensor:
    """
    I: (B,4,H,W) with order [0, π/2, π, 3π/2]
    return: (B,7,H,W) = [p0,p1,p2,p3, DC, Re, Im]
    """
    p0, p1, p2, p3 = I[:, 0:1], I[:, 1:2], I[:, 2:3], I[:, 3:4]
    DC = (p0 + p1 + p2 + p3) * 0.25
    Re = p0 - p2
    Im = p1 - p3
    return torch.cat([I, DC, Re, Im], dim=1)



class CorrelationSynthesizer(nn.Module):
    r"""
    y(t) = b + Σ_k [ a_k cos(ω_k t + φ) + s_k sin(ω_k t + φ) ]
    4-phase sampling: φ ∈ {0, π/2, π, 3π/2}
    """
    def __init__(self,
                 base_freq_hz: float = 50e6,
                 dc_baseline: float = 1.0,
                 use_soft_clip: bool = True,
                 soft_clip_temp: float = 6.0):
        super().__init__()
        freqs = build_freq_list(base_freq_hz)
        omega = 2.0 * math.pi * freqs
        self.register_buffer("freqs_hz", freqs)
        self.register_buffer("omega", omega)
        self.register_buffer("phase_shifts", torch.tensor(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=torch.float32))
        self.dc_baseline = float(dc_baseline)
        self.use_soft_clip = bool(use_soft_clip)
        self.soft_clip_temp = float(soft_clip_temp)


        self._noise_read = (0.001, 0.01)
        self._noise_shot = (0.0,   0.02)
        self._noise_offs = (-0.02, 0.02)
        self._noise_fpnr = (0.0,   0.005)
        self._noise_fpnc = (0.0,   0.005)
        self._noise_gain = (0.95, 1.05)
        self._noise_bits = 12

    def set_noise_ranges(self, read=None, shot=None, offs=None, fpnr=None, fpnc=None, gain=None, bits: int | None = None):
        if read is not None: self._noise_read = tuple(read)
        if shot is not None: self._noise_shot = tuple(shot)
        if offs is not None: self._noise_offs = tuple(offs)
        if fpnr is not None: self._noise_fpnr = tuple(fpnr)
        if fpnc is not None: self._noise_fpnc = tuple(fpnc)
        if gain is not None: self._noise_gain = tuple(gain)
        if bits is not None: self._noise_bits = int(bits)

    def _split_coeffs(self, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xk = xk.squeeze()
        if xk.ndim == 1: xk = xk.unsqueeze(0)
        B, D = xk.shape
        K = int(self.omega.numel())
        if D == K:
            a = xk; s = torch.zeros_like(a)
        elif D == 2*K:
            a, s = xk[:, :K], xk[:, K:]
        else:
            raise ValueError(f"bad xk dim={D}, expect K={K} or 2K={2*K}")
        return a, s

    def sample_map(self, xk: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        """Signal-only 4-phase: (B,4,H,W) in approx [0..2b]"""
        xk = xk.squeeze()
        if xk.ndim == 1: xk = xk.unsqueeze(0)
        a, s = self._split_coeffs(xk)
        if t_map.ndim == 2:
            t_map = t_map.unsqueeze(0).unsqueeze(0)
        elif t_map.ndim == 3:
            t_map = t_map.unsqueeze(1)
        elif t_map.ndim != 4:
            raise ValueError(f"bad t_map shape {tuple(t_map.shape)}")
        Bx, Bt = a.size(0), t_map.size(0)
        if Bx == 1 and Bt > 1:
            a = a.expand(Bt, -1); s = s.expand(Bt, -1)
        elif Bx != Bt:
            raise ValueError(f"batch mismatch xk({Bx}) vs t_map({Bt})")
        B, K = Bt, a.size(1)
        t32  = _f32(t_map).unsqueeze(1)
        w32  = _f32(self.omega[:K]).view(1, K, 1, 1, 1)
        ph32 = _f32(self.phase_shifts).view(1, 1, 4, 1, 1)
        a32, s32 = _f32(a).view(B, K, 1, 1, 1), _f32(s).view(B, K, 1, 1, 1)
        basis_c = torch.cos(w32 * t32 + ph32)
        basis_s = torch.sin(w32 * t32 + ph32)
        y = self.dc_baseline + (a32 * basis_c + s32 * basis_s).sum(dim=1)  # (B,4,H,W)
        if self.use_soft_clip:
            t = self.soft_clip_temp
            y = 2.0*self.dc_baseline*torch.sigmoid(t*(y/(2.0*self.dc_baseline)))
        return _nan_safe(y, 0.0, 2.0*self.dc_baseline)

    def sample_map_physics(self,
                           xk: torch.Tensor,
                           t_map: torch.Tensor,
                           alpha: Optional[torch.Tensor] = None,
                           ray_cos: Optional[torch.Tensor] = None,
                           cos_power: float = 2.0,
                           use_falloff: bool = True,
                           d0: float = 1.0) -> torch.Tensor:
        """
        out = (alpha * falloff * cos^eta) * y_signal   (NO env terms)
        """
        y_signal = self.sample_map(xk, t_map)
        B, _, H, W = y_signal.shape
        device, dtype = y_signal.device, y_signal.dtype

        if alpha is None:
            alpha = torch.ones((B,1,H,W), device=device, dtype=dtype)
        elif alpha.ndim == 3:
            alpha = alpha.unsqueeze(1)

        if use_falloff:
            if t_map.ndim == 4:
                z_approx = t_map * (_SPEED_OF_LIGHT / 2.0)
            elif t_map.ndim == 3:
                z_approx = t_map.unsqueeze(1) * (_SPEED_OF_LIGHT / 2.0)
            else:
                z_approx = t_map.unsqueeze(0).unsqueeze(0) * (_SPEED_OF_LIGHT / 2.0)
            z_approx = _nan_safe(z_approx.squeeze(1), 1e-6, 1e6)
            falloff = (d0 / z_approx).pow(2.0).clamp_max(1e6).view(B,1,H,W)
        else:
            falloff = 1.0

        if ray_cos is not None:
            if ray_cos.ndim == 2: ray_cos = ray_cos.view(1,1,H,W)
            elif ray_cos.ndim == 3: ray_cos = ray_cos.unsqueeze(1)
            angle = ray_cos.clamp(0.0, 1.0).pow(cos_power)
            if angle.size(0) == 1 and B > 1:
                angle = angle.expand(B, -1, -1, -1)
        else:
            angle = 1.0

        out = (alpha * falloff * angle) * y_signal
        return _nan_safe(out, 0.0, 2.0*self.dc_baseline)

    def to_unit(self, y: torch.Tensor) -> torch.Tensor:
        """Map [0..2b] -> [0..1]."""
        return (_nan_safe(y) / (2.0 * self.dc_baseline)).clamp(0.0, 1.0)


    def _paper_noise_unit(self, y01: torch.Tensor) -> torch.Tensor:
        x = _nan_safe(y01, 0.0, 1.0)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        def _rand(a, b):
            return torch.empty(1, device=device, dtype=torch.float32).uniform_(a, b).item()

        gain   = _rand(*self._noise_gain)
        offset = _rand(*self._noise_offs)
        y = x * gain + offset

        read_std = _rand(*self._noise_read)
        y = y + torch.randn_like(y, dtype=dtype) * read_std

        shot_k = _rand(*self._noise_shot)
        y = y + torch.randn_like(y, dtype=dtype) * torch.sqrt(_nan_safe(y, 0.0, 1e6)).to(dtype) * shot_k

        row_amp = _rand(*self._noise_fpnr)
        if row_amp > 0.0:
            y = y + torch.randn(B, 1, H, 1, device=device, dtype=dtype) * row_amp
        col_amp = _rand(*self._noise_fpnc)
        if col_amp > 0.0:
            y = y + torch.randn(B, 1, 1, W, device=device, dtype=dtype) * col_amp

        if self._noise_bits and self._noise_bits > 0:
            levels = (1 << int(self._noise_bits)) - 1
            y = torch.round(_nan_safe(y, 0.0, 1.0) * levels) / levels
        return _nan_safe(y, 0.0, 1.0)

    def paper_noise_like(self, x_raw: torch.Tensor, peak: float) -> torch.Tensor:
        if peak <= 0:
            return x_raw
        y01 = (_nan_safe(x_raw, 0.0, peak) / float(peak)).clamp(0.0, 1.0)
        y01 = self._paper_noise_unit(y01)
        return _nan_safe(y01 * float(peak), 0.0, peak)


# -------------------- blocks --------------------
class MambaBlock2D(nn.Module):
    def __init__(self, dim: int, use_mamba: bool = True, force_fp32: bool = True,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.use_mamba = bool(use_mamba and _HAS_MAMBA)
        self.force_fp32 = bool(force_fp32)
        if self.use_mamba:
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
                nn.Conv2d(dim, dim, 3, padding=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_mamba:
            return x + self.block(x)

        
        ctx = contextlib.nullcontext()
        if x.is_cuda and self.force_fp32:
            ctx = torch.autocast(device_type='cuda', enabled=False)

        with ctx:
            in_dtype = x.dtype
            B, C, H, W = x.shape
            y = x.to(torch.float32).permute(0, 2, 3, 1).reshape(B, H * W, C)
            y = self.norm(y)
            y = self.mamba(y) + y
            y = self.ffn(y) + y
            y = y.reshape(B, H, W, C).permute(0, 3, 1, 2).to(in_dtype)
        return y

class SE(nn.Module):
    def __init__(self, dim: int, r: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1), nn.GELU(),
            nn.Conv2d(dim // r, dim, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.net(x); return x * w

class Down(nn.Module):
    def __init__(self, c_in, c_out, use_mamba=True, force_fp32=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU(),
            SE(c_out),
        )
    def forward(self, x): return self.op(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out, use_mamba=True, force_fp32=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU(),
            SE(c_out),
        )
    def forward(self, x): return self.op(x)



class BackboneUShape(nn.Module):
    """
    in: 7ch (PhaseMix) or 4ch (raw phases)
    out: full-res feat for depth head; pooled feat for waveform head
    """
    def __init__(self, in_ch: int = 7, dim: int = 160, n1: int = 2, n2: int = 4, n3: int = 6,
                 use_skip: bool = True, use_mamba: bool = True, mamba_force_fp32: bool = True):
        super().__init__()
        self.use_skip = bool(use_skip)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 7, padding=3), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )
        self.enc1 = nn.Sequential(*[MambaBlock2D(dim, use_mamba, mamba_force_fp32) for _ in range(n1)])
        self.down1 = Down(dim, dim*2, use_mamba, mamba_force_fp32)
        self.enc2 = nn.Sequential(*[MambaBlock2D(dim*2, use_mamba, mamba_force_fp32) for _ in range(n2)])
        self.down2 = Down(dim*2, dim*4, use_mamba, mamba_force_fp32)
        self.bott = nn.Sequential(*[MambaBlock2D(dim*4, use_mamba, mamba_force_fp32) for _ in range(n3)])
        self.up2  = Up(dim*4, dim*2, use_mamba, mamba_force_fp32)
        self.dec2 = nn.Sequential(*[MambaBlock2D(dim*2, use_mamba, mamba_force_fp32) for _ in range(max(1, n2//2))])
        self.up1  = Up(dim*2, dim, use_mamba, mamba_force_fp32)
        self.dec1 = nn.Sequential(*[MambaBlock2D(dim, use_mamba, mamba_force_fp32) for _ in range(max(1, n1//2))])

        self.depth_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, 1, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.wave_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(),
            nn.Conv2d(dim, 30, 1)  # 2K=30
        )

    def forward(self, x_in: torch.Tensor):
        x = self.stem(x_in)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b  = self.bott(self.down2(e2))
        u2 = self.up2(b)
        d2 = self.dec2(u2 + e2 if self.use_skip else u2)
        u1 = self.up1(d2)
        d1 = self.dec1(u1 + e1 if self.use_skip else u1)

        depth_raw = self.depth_head(d1)             # (B,1,H,W)
        xk_raw = self.wave_head(self.gap(d1)).flatten(1)  # (B,30)
        return d1, depth_raw, xk_raw



class RGBAlbedoEstimator(nn.Module):
    def __init__(self, dim: int = 64, alpha_min: float = 0.2, alpha_max: float = 1.5, srgb_gamma: float = 2.2):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.srgb_gamma = float(srgb_gamma)
        self.net = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1)
        )

    def _srgb_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x.pow(self.srgb_gamma)

    def forward(self, rgb: torch.Tensor, rgb_is_srgb: bool = True) -> torch.Tensor:
        if rgb_is_srgb:
            x = self._srgb_to_linear(rgb)
        else:
            x = rgb.clamp(0.0, 1.0)
        a = torch.sigmoid(self.net(x))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * a
        return _nan_safe(alpha, self.alpha_min, self.alpha_max)



class PhaseMambaNet(nn.Module):
    def __init__(self,
                 dim: int = 160,
                 n1: int = 2, n2: int = 4, n3: int = 6,
                 base_freq_hz: float = 50e6,
                 dmax: float = 8.0,
                 use_falloff: bool = True, cos_power: float = 2.0,
                 depth_is_z: bool = True,
                 use_rgb_albedo: bool = True,
                 albedo_dim: int = 64, albedo_min: float = 0.2, albedo_max: float = 1.5,
                 use_phasemix: bool = True,
                 use_skip: bool = True,
                 use_mamba: bool = True,
                 mamba_force_fp32: bool = True):
        super().__init__()
        self.dmax = float(dmax)
        self.use_falloff = bool(use_falloff)
        self.cos_power = float(cos_power)
        self.depth_is_z = bool(depth_is_z)
        self.use_rgb_albedo = bool(use_rgb_albedo)
        self.use_phasemix = bool(use_phasemix)

        
        self.corr = CorrelationSynthesizer(base_freq_hz=base_freq_hz, dc_baseline=1.0)

        # backbone
        in_ch = 7 if self.use_phasemix else 4
        self.backbone = BackboneUShape(in_ch=in_ch, dim=dim, n1=n1, n2=n2, n3=n3,
                                       use_skip=use_skip, use_mamba=use_mamba, mamba_force_fp32=mamba_force_fp32)

        
        self.k_freq = int(build_freq_list(base_freq_hz).numel())   
        self.k_param = 2 * self.k_freq                             
        self.register_buffer("xk_min", torch.zeros(1, self.k_param))
        self.register_buffer("xk_max", torch.zeros(1, self.k_param))
        self.global_xk_logits = nn.Parameter(torch.zeros(self.k_param))

        
        self._init_default_waveform_box(spread=0.8)

        # albedo
        if self.use_rgb_albedo:
            self.albedo_net = RGBAlbedoEstimator(dim=albedo_dim, alpha_min=albedo_min, alpha_max=albedo_max)
        else:
            self.albedo_net = None

    
    @torch.no_grad()
    def _init_default_waveform_box(self, spread: float = 0.8):
        
        device = self.xk_min.device
        K = self.k_freq
        
        per = float(spread) / max(K, 1)
        a_min = -per * torch.ones(K, device=device)
        a_max =  per * torch.ones(K, device=device)
        s_min = -per * torch.ones(K, device=device)
        s_max =  per * torch.ones(K, device=device)
        self.xk_min.copy_(torch.cat([a_min, s_min])[None])
        self.xk_max.copy_(torch.cat([a_max, s_max])[None])
        self.global_xk_logits.zero_()

    @torch.no_grad()
    def set_waveform_box(self, xk_min: torch.Tensor, xk_max: torch.Tensor):
        xk_min = torch.as_tensor(xk_min, dtype=torch.float32, device=self.xk_min.device).flatten()
        xk_max = torch.as_tensor(xk_max, dtype=torch.float32, device=self.xk_max.device).flatten()
        if xk_min.numel() == self.k_freq and xk_max.numel() == self.k_freq:
            xk_min = torch.cat([xk_min, xk_min], dim=0)
            xk_max = torch.cat([xk_max, xk_max], dim=0)
        assert xk_min.numel() == xk_max.numel() == 2 * self.k_freq, "xk_min/xk_max 维度必须为 2K"

        
        same = (xk_max <= xk_min)
        if same.any():
            delta = 1e-3
            xk_min = torch.where(same, xk_min - delta, xk_min)
            xk_max = torch.where(same, xk_max + delta, xk_max)

        self.xk_min.copy_(xk_min.view(1, -1))
        self.xk_max.copy_(xk_max.view(1, -1))

        mid = 0.5 * (self.xk_min + self.xk_max)
        rng = (self.xk_max - self.xk_min).clamp_min(1e-6)
        init = (mid - self.xk_min) / rng
        init = init.clamp(1e-4, 1.0 - 1e-4)
        self.global_xk_logits.copy_(torch.logit(init).squeeze())

    def _xk_from_post_box(self) -> torch.Tensor:
        xk_sig = torch.sigmoid(self.global_xk_logits).unsqueeze(0)
        rng = self.xk_max - self.xk_min
        tiny = rng.abs() < 1e-9
        if tiny.any():
            widen = 1e-3
            xk_min = torch.where(tiny, self.xk_min - widen, self.xk_min)
            xk_max = torch.where(tiny, self.xk_max + widen, self.xk_max)
            rng = xk_max - xk_min
        else:
            xk_min, xk_max = self.xk_min, self.xk_max
        xk = xk_min + rng * xk_sig
        return xk

    def get_global_xk(self) -> torch.Tensor:
        return self._xk_from_post_box()

    def sanitize_(self):
        with torch.no_grad():
            self.global_xk_logits.data.clamp_(-8.0, 8.0)
            self.global_xk_logits.data = torch.nan_to_num(
                self.global_xk_logits.data, nan=0.0, posinf=8.0, neginf=-8.0)

    
    def forward_from_depth_train(self, gt_depth: torch.Tensor,
                                 rgb: Optional[torch.Tensor],
                                 intrinsics: Optional[Union[torch.Tensor, Tuple[float,float,float,float], dict]],
                                 to_unit: bool = True, rgb_is_srgb: bool = True):
        
        print("gt_depth[m]:", gt_depth.min().item(), gt_depth.mean().item(), gt_depth.max().item())



        
        if torch.allclose(self.xk_min, self.xk_max):
            self._init_default_waveform_box(spread=0.8)

        device = next(self.parameters()).device
        gt_depth = _nan_safe(gt_depth.to(device), 0.0, self.dmax)
        B, _, H, W = gt_depth.shape

        # albedo
        if (self.use_rgb_albedo and (rgb is not None)):
            rgb = rgb.to(device, dtype=torch.float32)
            if rgb.shape[-2:] != (H, W):
                raise ValueError(f"RGB size {rgb.shape[-2:]} != depth size {(H,W)}")
            alpha_map = self.albedo_net(rgb, rgb_is_srgb=rgb_is_srgb)  # (B,1,H,W)
        else:
            alpha_map = torch.ones((B,1,H,W), device=device, dtype=torch.float32)

        
        ray_cos = None
        if intrinsics is not None:
            ray_cos = cosine_from_intrinsics(H, W, intrinsics, device, dtype=gt_depth.dtype)

        
        if intrinsics is not None and self.depth_is_z:
            if torch.is_tensor(intrinsics) and intrinsics.numel() == 9:
                t_map = depth_to_phase_t_intrinsics(gt_depth, K=intrinsics, depth_is_z=True)
            elif torch.is_tensor(intrinsics) and intrinsics.numel() == 4:
                fx, fy, cx, cy = [float(v) for v in intrinsics.view(-1)]
                t_map = depth_to_phase_t_intrinsics(gt_depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            elif isinstance(intrinsics, (tuple, list)):
                fx, fy, cx, cy = [float(v) for v in intrinsics]
                t_map = depth_to_phase_t_intrinsics(gt_depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            elif isinstance(intrinsics, dict):
                fx, fy, cx, cy = float(intrinsics["fx"]), float(intrinsics["fy"]), float(intrinsics["cx"]), float(intrinsics["cy"])
                t_map = depth_to_phase_t_intrinsics(gt_depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            else:
                t_map = depth_to_phase_t(gt_depth)
        else:
            t_map = depth_to_phase_t(gt_depth)

        
        xk = self.get_global_xk().to(device)
        xkB = xk.expand(B, -1)

        
        I_raw = self.corr.sample_map_physics(
            xkB, t_map, alpha=alpha_map, ray_cos=ray_cos,
            cos_power=self.cos_power, use_falloff=self.use_falloff
        )

        
        if self.training:
            peak = 2.0 * self.corr.dc_baseline
            I_raw = self.corr.paper_noise_like(I_raw, peak=peak)
        I_obs01 = self.corr.to_unit(I_raw) if to_unit else I_raw

        
        x_in = phasemix_cat(I_obs01) if self.use_phasemix else I_obs01

        
        _, depth_raw, _ = self.backbone(x_in)
        depth = (torch.tanh(depth_raw.clamp(-3, 3)) * 0.5 + 0.5) * self.dmax
        depth = _nan_safe(depth, 0.0, self.dmax)

        
        if intrinsics is not None and self.depth_is_z:
            if torch.is_tensor(intrinsics) and intrinsics.numel() == 9:
                t_pred = depth_to_phase_t_intrinsics(depth, K=intrinsics, depth_is_z=True)
            elif torch.is_tensor(intrinsics) and intrinsics.numel() == 4:
                fx, fy, cx, cy = [float(v) for v in intrinsics.view(-1)]
                t_pred = depth_to_phase_t_intrinsics(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            elif isinstance(intrinsics, (tuple, list)):
                fx, fy, cx, cy = [float(v) for v in intrinsics]
                t_pred = depth_to_phase_t_intrinsics(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            elif isinstance(intrinsics, dict):
                fx, fy, cx, cy = float(intrinsics["fx"]), float(intrinsics["fy"]), float(intrinsics["cx"]), float(intrinsics["cy"])
                t_pred = depth_to_phase_t_intrinsics(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_z=True)
            else:
                t_pred = depth_to_phase_t(depth)
        else:
            t_pred = depth_to_phase_t(depth)

        pred4 = self.corr.sample_map_physics(
            xkB, t_pred,
            alpha=alpha_map, ray_cos=ray_cos, cos_power=self.cos_power, use_falloff=self.use_falloff
        )
        pred4_01 = self.corr.to_unit(pred4)

        extras = {
            "I_obs01": _nan_safe(I_obs01, 0.0, 1.0),
            "I_pred01": _nan_safe(pred4_01, 0.0, 1.0),
            "alpha_map": alpha_map,
        }
        return xkB, depth, extras

    
    @torch.no_grad()
    def infer_from_four_phase(self, I_obs01: torch.Tensor):
    
        self.eval()
        device = next(self.parameters()).device
        I_obs01 = I_obs01.to(device, dtype=torch.float32).clamp(0.0, 1.0)
        x_in = phasemix_cat(I_obs01) if self.use_phasemix else I_obs01
        _, depth_raw, _ = self.backbone(x_in)
        depth = (torch.tanh(depth_raw.clamp(-3, 3)) * 0.5 + 0.5) * self.dmax
        xk = self.get_global_xk().expand(I_obs01.size(0), -1)
        return xk, depth



class PhaseMambaPhysLoss(nn.Module):
    def __init__(self,
                 mae_w: float = 1.0,
                 ssim_w: float = 0.5,
                 zncc_w: float = 0.25,
                 ssim_max_val: float = 1.0,
                 dmax_for_ssim: float = 8.0,
                 base_freq_hz: float = 50e6,
                 freq_l1_w: float = 1e-3,
                 freq_alpha: float = 1.0,
                 ms_scales: Optional[List[int]] = None,
                 ms_weights: Optional[List[float]] = None,
                 **kwargs):
        super().__init__()
        self.mae_w = mae_w
        self.ssim_w = ssim_w
        self.zncc_w = zncc_w
        self.ssim_max_val = ssim_max_val
        self.dmax_for_ssim = dmax_for_ssim
        self.freq_l1_w = float(freq_l1_w)

        freqs = build_freq_list(base_freq_hz)
        w = (freqs.float() / freqs.max()).pow(freq_alpha)
        self.register_buffer("freq_weights", w)

        if ms_scales is None:
            ms_scales = (kwargs.get("ms_zncc_scales")
                         or kwargs.get("zncc_scales")
                         or [1])
        if ms_weights is None:
            ms_weights = (kwargs.get("ms_zncc_weights")
                          or kwargs.get("zncc_weights")
                          or [1.0])

        if isinstance(ms_scales, str):
            ms_scales = [int(x) for x in ms_scales.split(",") if x.strip() != ""]
        if isinstance(ms_weights, str):
            ms_weights = [float(x) for x in ms_weights.split(",") if x.strip() != ""]

        if len(ms_scales) != len(ms_weights):
            L = min(len(ms_scales), len(ms_weights))
            ms_scales, ms_weights = ms_scales[:L], ms_weights[:L]
        if len(ms_scales) == 0:
            ms_scales, ms_weights = [1], [1.0]

        self.ms_scales  = [int(s) for s in ms_scales]
        self.ms_weights = [float(v) for v in ms_weights]

    @staticmethod
    def zncc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        b = torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0)
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        a_std = a.std(dim=1, keepdim=True) + eps
        b_std = b.std(dim=1, keepdim=True) + eps
        corr = ((a * b).sum(dim=1) / (a_std * b_std)).clamp(-1, 1)
        return corr.mean()

    def _ms_zncc(self, pred01: torch.Tensor, obs01: torch.Tensor) -> torch.Tensor:
        zs = []
        Wsum = 0.0
        for s, w in zip(self.ms_scales, self.ms_weights):
            if s <= 1:
                pp, oo = pred01, obs01
            else:
                pp = F.avg_pool2d(pred01, kernel_size=s, stride=s)
                oo = F.avg_pool2d(obs01,  kernel_size=s, stride=s)
            z = self.zncc(pp, oo)
            zs.append(w * (1.0 - z))
            Wsum += w
        return sum(zs) / max(Wsum, 1e-6)

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
                pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                pred_xk: torch.Tensor,
                obs_four_phase01: torch.Tensor, pred_four_phase01: torch.Tensor):
        device = pred_depth.device
        gt_depth = gt_depth.to(device)

        # MAE
        mae = F.l1_loss(pred_depth, gt_depth)


        pd = (pred_depth / max(1e-6, self.dmax_for_ssim)).clamp(0, 1)
        gd = (gt_depth   / max(1e-6, self.dmax_for_ssim)).clamp(0, 1)
        ssim_val = ssim(pd, gd, max_val=self.ssim_max_val).clamp(0.0, 1.0)
        ssim_loss = 1.0 - ssim_val


        zncc_loss = self._ms_zncc(pred_four_phase01, obs_four_phase01)


        freq_prior = self._freq_l1_prior(pred_xk)

        total = (self.mae_w * mae + self.ssim_w * ssim_loss +
                 self.zncc_w * zncc_loss + self.freq_l1_w * freq_prior)

        logs = {
            "mae": float(mae.detach()),
            "ssim": float(ssim_val.detach()),
            "zncc": float(1.0 - zncc_loss.detach()),
            "freq_prior": float(freq_prior.detach()),
            "total": float(total.detach()),
        }
        return total, logs
