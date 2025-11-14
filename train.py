import os, re, argparse, random, json, shutil
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import PhaseMambaNet, PhaseMambaPhysLoss

# ---------------- I/O ----------------
def read_depth_image(path: Union[str, Path], depth_scale: Optional[float]) -> torch.Tensor:

    im = Image.open(path)
    arr = np.array(im)
    if arr.dtype == np.uint16:
        scale = 1000.0 if depth_scale is None else float(depth_scale)
        depth = torch.from_numpy(arr.astype(np.float32)) / scale
    elif arr.dtype == np.uint8:
        if depth_scale is None:
            raise ValueError(f"{path} is 8-bit; please set --depth_scale")
        depth = torch.from_numpy(arr.astype(np.float32)) / float(depth_scale)
    else:

        depth = torch.from_numpy(arr.astype(np.float32))
    return depth  # (H,W), meters

def read_rgb_image(path: Union[str, Path]) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    arr = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W) 0..1

def load_intrinsics_tuple(args):
    if args.K_txt is not None:
        p = Path(args.K_txt)
        if p.suffix.lower() == ".npy":
            K = np.load(str(p)).astype(np.float32).reshape(3,3)
        else:
            K = np.loadtxt(str(p)).astype(np.float32).reshape(3,3)
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        return (fx, fy, cx, cy)
    return (float(args.fx), float(args.fy), float(args.cx), float(args.cy))

def scale_intrinsics(intr: Tuple[float,float,float,float], sx: float, sy: float):
    fx, fy, cx, cy = intr
    return (fx * sx, fy * sy, cx * sx, cy * sy)

# --------------- Dataset ---------------
class DepthRGBDataset(Dataset):
    def __init__(self, gt_dir: Union[str, Path], rgb_dir: Union[str, Path],
                 indices: List[int], depth_scale: Optional[float] = 1000.0,
                 rgb_tmpl: str = "image_{:04d}.png", rgb_policy: str = "error"):
        self.gt_dir = Path(gt_dir)
        self.rgb_dir = Path(rgb_dir)
        self.indices = indices
        self.depth_scale = depth_scale
        self.gt_tmpl = "depth_{:04d}.png"
        self.rgb_tmpl = rgb_tmpl
        self.rgb_policy = rgb_policy

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        n = self.indices[i]
        gt_path = self.gt_dir / self.gt_tmpl.format(n)
        if not gt_path.exists():
            raise FileNotFoundError(f"Depth not found: {gt_path}")
        gt_depth = read_depth_image(str(gt_path), self.depth_scale)  # (H,W), meters

        rgb_paths = [
            self.rgb_dir / self.rgb_tmpl.format(n),
            (self.rgb_dir / self.rgb_tmpl.format(n)).with_suffix(".jpg"),
            (self.rgb_dir / self.rgb_tmpl.format(n)).with_suffix(".jpeg")
        ]
        rgb_path = next((p for p in rgb_paths if p.exists()), None)

        if rgb_path is None:
            if self.rgb_policy == "error":
                raise FileNotFoundError(f"RGB not found: {rgb_paths[0]}")
            elif self.rgb_policy == "skip":
                return None
            else:
                rgb = torch.zeros(3, *gt_depth.shape, dtype=torch.float32)
        else:
            rgb = read_rgb_image(str(rgb_path)).float()
            if rgb.shape[1:] != gt_depth.shape:
                rgb = F.interpolate(rgb.unsqueeze(0), size=gt_depth.shape, mode='bilinear', align_corners=False).squeeze(0)

        return {
            "gt_depth": gt_depth.float().unsqueeze(0),   # (1,H,W), meters
            "rgb": rgb,                                   # (3,H,W), 0..1
            "index": n,
            "rgb_path": str(rgb_path) if rgb_path is not None else None
        }

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# --------------- split helpers ---------------
def list_indices(gt_dir: Union[str, Path]) -> List[int]:
    gt_dir = Path(gt_dir)
    pat = re.compile(r"depth_(\d+)\.png")
    idxs = []
    for fn in os.listdir(gt_dir):
        m = pat.match(fn)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)

def build_splits_depth(gt_dir: Union[str, Path]) -> Tuple[List[int], List[int]]:

    idxs_s = list_indices(gt_dir)
    k = min(1000, len(idxs_s))
    return idxs_s[:k], idxs_s[k:]

# --------------- utils ---------------
def param_checksum(model) -> float:
    with torch.no_grad():
        s = 0.0
        for p in model.parameters():
            if p.requires_grad:
                s += float(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).abs().sum())
        return s

def tensor_stats(x: torch.Tensor) -> str:
    x = x.detach().float()
    return f"min={float(x.min()):.4f} mean={float(x.mean()):.4f} max={float(x.max()):.4f}"

def save_depth_16bit_and_rgb(out_dir: Path, depth_m: torch.Tensor, index: int, max_depth_vis: float = 10.0):

    out_dir.mkdir(parents=True, exist_ok=True)
    d = depth_m.detach().cpu().float().clamp(0, max_depth_vis)   # (H,W), meters
    d16 = (d / max_depth_vis * 65535.0).round().clamp(0, 65535).to(torch.uint16).numpy()
    Image.fromarray(d16).save(out_dir / f"depth16_{index:04d}.png")
    # 简单伪彩
    d01 = (d / max_depth_vis).numpy()
    rgb = np.zeros((*d01.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * np.clip(4 * (d01 - 0.25), 0, 1)).astype(np.uint8)
    rgb[..., 1] = (255 * np.clip(4 * np.abs(d01 - 0.5) - 1, 0, 1) * 0.75).astype(np.uint8)
    rgb[..., 2] = (255 * np.clip(1.0 - 4 * np.abs(d01 - 0.75), 0, 1)).astype(np.uint8)
    Image.fromarray(rgb).save(out_dir / f"depth16_rgb_{index:04d}.png")

def save_depth_16bit_mm(out_dir: Path, depth_m: torch.Tensor, index: int):

    out_dir.mkdir(parents=True, exist_ok=True)
    d_mm = (depth_m.detach().cpu().float().clamp(min=0) * 1000.0)
    d16 = d_mm.round().clamp(0, 65535).to(torch.uint16).numpy()
    Image.fromarray(d16).save(out_dir / f"depth16mm_{index:04d}.png")

def save_xk_global(out_dir: Path, freqs_hz: torch.Tensor, xk: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = [float(v) for v in freqs_hz.detach().cpu().tolist()]
    xk = xk.detach().cpu().mean(dim=0).float()
    K = len(freqs)
    if xk.numel() == 2*K:
        a, s = xk[:K], xk[K:]
    elif xk.numel() == K:
        a, s = xk, torch.zeros_like(xk)
    else:
        raise ValueError(f"xk length {xk.numel()} not in {{K,2K}}")
    l1 = (a.abs() + s.abs()).sum().clamp(min=1e-12)
    a_n = (a / l1).tolist(); s_n = (s / l1).tolist()
    with open(out_dir / "xk_global.json", "w") as f:
        json.dump({"freqs_hz": freqs, "a_l1norm": a_n, "s_l1norm": s_n, "dc_baseline": 1.0}, f, indent=2)

# ---------------- DDP helpers ----------------
def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        is_dist = True
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_dist = False
    return is_dist, rank, world_size, local_rank

def cleanup_distributed_mode():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

def unwrap(model):
    return model.module if isinstance(model, DDP) else model

# --------------- Early Stopping ---------------
class EarlyStopper:
    def __init__(self, metric_name: str = "val_mae", patience: int = 20, min_delta: float = 1e-3, warmup_epochs: int = 10):

        self.metric_name = metric_name
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best = float("inf")
        self.no_improve = 0

    def update(self, epoch: int, val_metric: float):
        improved = False
        if val_metric + self.min_delta < self.best:
            self.best = float(val_metric)
            self.no_improve = 0
            improved = True
        else:
            if epoch > self.warmup_epochs:
                self.no_improve += 1
        should_stop = (self.no_improve >= self.patience)
        return improved, should_stop, self.best, self.no_improve

# --------------- main ---------------
def parse_size_arg(s: Optional[str]):
    if s is None or s.lower() in ["", "none", "0"]:
        return None
    if "x" in s:
        w, h = s.split("x")
        return int(h), int(w)  # (H,W)
    raise ValueError("--train_size format must be WxH, e.g., 320x240")

def parse_list(s: str, cast):
    if s is None: return []
    if s.strip() == "": return []
    return [cast(x) for x in s.split(",")]

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if torch.is_floating_point(v)}
        self.backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and torch.is_floating_point(v):
                self.shadow[k].mul_((self.decay)).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: torch.nn.Module):
        if self.backup is not None:
            model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
            self.backup = None

def main():
    parser = argparse.ArgumentParser()

    # ---------- Paths ----------
    parser.add_argument("--gt_dir", type=str, default="./nyu_output/depths")
    parser.add_argument("--rgb_dir", type=str, default="./nyu_output/images")
    parser.add_argument("--rgb_tmpl", type=str, default="image_{:04d}.png")
    parser.add_argument("--rgb_policy", type=str, default="error", choices=["error", "skip", "zero"])
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="uint16 的单位缩放；GT 为毫米时设 1000")

    # ---------- Intrinsics ----------
    parser.add_argument("--K_txt", type=str, default=None)
    parser.add_argument("--fx", type=float, default=516.252)
    parser.add_argument("--fy", type=float, default=516.432)
    parser.add_argument("--cx", type=float, default=318.615)
    parser.add_argument("--cy", type=float, default=237.579)
    parser.add_argument("--depth_is_z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_falloff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cos_power", type=float, default=2.0)

    # ---------- Training ----------
    parser.add_argument("--epochs", type=int, default=150, help="最大训练轮数（上限）；早停满足则提前退出")
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_wave", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="runs/nyu_phaseclean")
    parser.add_argument("--clean_save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16","bf16"])
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--blocks_enc", type=str, default="2,4,6")  # n1,n2,n3
    parser.add_argument("--dmax", type=float, default=10.0)
    parser.add_argument("--base_freq_hz", type=float, default=50e6)
    parser.add_argument("--train_size", type=str, default=None, help="training-only resize, e.g., 320x240")

    # ---------- Architecture toggles ----------
    parser.add_argument("--use_mamba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mamba_force_fp32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_phasemix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_skip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_rgb_albedo", action=argparse.BooleanOptionalAction, default=False)

    # ---------- LR Scheduler ----------
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-6)

    # ---------- Early stopping ----------
    parser.add_argument("--early_stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--es_metric", type=str, default="val_mae", choices=["val_mae","val_loss"])
    parser.add_argument("--es_patience", type=int, default=20)
    parser.add_argument("--es_min_delta", type=float, default=1e-3)
    parser.add_argument("--es_warmup_epochs", type=int, default=10)

    # ---------- Tricks ----------
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--clip_grad", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--tta_flip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_pbar", action=argparse.BooleanOptionalAction, default=False)

    # ---------- Noise ----------
    parser.add_argument("--no_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise_read", nargs=2, type=float, default=[0.001, 0.01])
    parser.add_argument("--noise_shot", nargs=2, type=float, default=[0.0, 0.02])
    parser.add_argument("--noise_gain", nargs=2, type=float, default=[0.95, 1.05])
    parser.add_argument("--noise_offs", nargs=2, type=float, default=[-0.02, 0.02])
    parser.add_argument("--noise_fpnr", nargs=2, type=float, default=[0.0, 0.005])
    parser.add_argument("--noise_fpnc", nargs=2, type=float, default=[0.0, 0.005])
    parser.add_argument("--noise_bits", type=int, default=12)
    parser.add_argument("--open_noise_epoch", type=int, default=10)
    parser.add_argument("--noise_ramp_epochs", type=int, default=10)

    # ---------- Loss weights + MS-ZNCC ----------
    parser.add_argument("--zncc_w", type=float, default=0.25)
    parser.add_argument("--ssim_w", type=float, default=0.5)
    parser.add_argument("--freq_l1_w", type=float, default=1e-3)
    parser.add_argument("--freq_alpha", type=float, default=1.0)
    parser.add_argument("--ms_zncc_scales", type=str, default="1,2,4")
    parser.add_argument("--ms_zncc_weights", type=str, default="1.0,0.5,0.25")

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

# 宽容解析：忽略未知参数而不中断
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Ignoring unknown args:", unknown)

    # DDP init
    is_dist, rank, world_size, local_rank = init_distributed_mode()
    is_main = (rank == 0)
    if is_dist:
        args.device = f"cuda:{local_rank}"

    # Repro
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; fallback to CPU.")
        args.device = "cpu"
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank); random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    save_dir = Path(args.save_dir)
    if is_main:
        if args.clean_save and save_dir.exists(): shutil.rmtree(save_dir)
        (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (save_dir / "final").mkdir(parents=True, exist_ok=True)

    # splits & datasets（固定前1000为train）
    train_idx, test_idx = build_splits_depth(args.gt_dir)
    if is_main:
        print(f"[from_depth] Train: {len(train_idx)}, Test: {len(test_idx)}")

    train_ds = DepthRGBDataset(args.gt_dir, args.rgb_dir, train_idx, depth_scale=args.depth_scale,
                               rgb_tmpl=args.rgb_tmpl, rgb_policy=args.rgb_policy)
    test_ds  = DepthRGBDataset(args.gt_dir, args.rgb_dir, test_idx,  depth_scale=args.depth_scale,
                               rgb_tmpl=args.rgb_tmpl, rgb_policy="skip")

    # samplers / loaders
    pin = args.device.startswith("cuda")
    if is_dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=False)
        test_sampler  = torch.utils.data.distributed.DistributedSampler(test_ds,  shuffle=False, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=False, collate_fn=_collate_skip_none)
        test_loader  = DataLoader(test_ds, batch_size=args.bs, sampler=test_sampler,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=False, collate_fn=_collate_skip_none)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=False, collate_fn=_collate_skip_none)
        test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=False, collate_fn=_collate_skip_none)

    # intrinsics
    intrinsics_base = load_intrinsics_tuple(args)
    if is_main:
        print(f"[Info] Intrinsics: fx={intrinsics_base[0]:.3f}, fy={intrinsics_base[1]:.3f}, "
              f"cx={intrinsics_base[2]:.3f}, cy={intrinsics_base[3]:.3f}")


    cos_min = torch.tensor([
        
    ], dtype=torch.float32)
    cos_max = torch.tensor([
        
    ], dtype=torch.float32)
    sin_min = torch.tensor([
        
    ], dtype=torch.float32)
    sin_max = torch.tensor([
        
    ], dtype=torch.float32)
    xk_min = torch.cat([cos_min, sin_min], dim=0)
    xk_max = torch.cat([cos_max, sin_max], dim=0)

    # model
    n1, n2, n3 = [int(x) for x in args.blocks_enc.split(",")]
    model = PhaseMambaNet(
        dim=args.dim, n1=n1, n2=n2, n3=n3,
        base_freq_hz=args.base_freq_hz, dmax=args.dmax,
        use_falloff=args.use_falloff, cos_power=args.cos_power,
        depth_is_z=args.depth_is_z,
        use_rgb_albedo=not args.no_rgb_albedo,
        use_phasemix=args.use_phasemix,
        use_skip=not args.no_skip,
        use_mamba=args.use_mamba,
        mamba_force_fp32=args.mamba_force_fp32
    ).to(args.device)

    model.set_waveform_box(xk_min, xk_max)

    # wrap DDP
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # noise
    unwrap(model).corr.set_noise_ranges(
        read=args.noise_read, shot=args.noise_shot, offs=args.noise_offs,
        fpnr=args.noise_fpnr, fpnc=args.noise_fpnc, gain=args.noise_gain, bits=args.noise_bits
    )
    if args.no_noise:
        unwrap(model).corr.set_noise_ranges(read=(0,0), shot=(0,0), offs=(0,0),
                                            fpnr=(0,0), fpnc=(0,0), gain=(1,1), bits=0)
        if is_main: print("[Info] Noise disabled.")

    # loss (multi-scale ZNCC)
    ms_scales  = parse_list(args.ms_zncc_scales, int)
    ms_weights = parse_list(args.ms_zncc_weights, float)
    loss_fn = PhaseMambaPhysLoss(
        mae_w=1.0, ssim_w=args.ssim_w, zncc_w=args.zncc_w,
        ssim_max_val=1.0, dmax_for_ssim=args.dmax,
        base_freq_hz=args.base_freq_hz,
        freq_l1_w=args.freq_l1_w, freq_alpha=args.freq_alpha,
        ms_scales=ms_scales, ms_weights=ms_weights
    ).to(args.device)
    if is_main:
        print(f"[MS-ZNCC] scales={ms_scales}, weights={ms_weights}")

    # optimizer
    base_params, wave_params = [], []
    for n, p in unwrap(model).named_parameters():
        if n == "global_xk_logits":
            wave_params.append(p)
        else:
            base_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": wave_params, "lr": args.lr_wave, "weight_decay": 0.0},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=args.lr_factor, patience=args.patience,
                                  min_lr=args.lr_min, verbose=is_main)

    # AMP/EMA
    use_amp = args.amp and args.device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    use_ema = args.ema
    ema = EMA(unwrap(model), decay=args.ema_decay) if use_ema else None

    # Early Stopper
    es = EarlyStopper(metric_name=args.es_metric, patience=args.es_patience,
                      min_delta=args.es_min_delta, warmup_epochs=args.es_warmup_epochs) if args.early_stop else None

    best_val = float("inf")
    best_mae = float("inf")
    if is_main:
        best_ckpt = Path(args.save_dir) / "checkpoints" / "best.pt"

    # training resize
    train_size = parse_size_arg(args.train_size)  # (H,W) or None

    def _finite_or_skip(total_loss, batch_dbg: str) -> bool:
        if not torch.isfinite(total_loss):
            if is_main: print(f"[skip batch] non-finite loss @ {batch_dbg}")
            return False
        return True

    should_break_all = False

    for epoch in range(1, args.epochs + 1):
        if is_dist:
            train_loader.sampler.set_epoch(epoch)

        # noise ramp
        def _scale_rng(rng, s):
            lo, hi = float(rng[0]), float(rng[1])
            return (lo * s, hi * s)
        if not args.no_noise:
            if epoch < args.open_noise_epoch:
                unwrap(model).corr.set_noise_ranges(read=(0,0), shot=(0,0), offs=(0,0),
                                                    fpnr=(0,0), fpnc=(0,0), gain=(1,1), bits=0)
            else:
                r = min(1.0, (epoch - args.open_noise_epoch + 1) / max(1, args.noise_ramp_epochs))
                unwrap(model).corr.set_noise_ranges(
                    read=_scale_rng(args.noise_read, r),
                    shot=_scale_rng(args.noise_shot, r),
                    offs=_scale_rng(args.noise_offs, r),
                    fpnr=_scale_rng(args.noise_fpnr, r),
                    fpnc=_scale_rng(args.noise_fpnc, r),
                    gain=(1 + (args.noise_gain[0] - 1) * r, 1 + (args.noise_gain[1] - 1) * r),
                    bits=args.noise_bits if r >= 0.7 else 0
                )

        # warmup
        if epoch <= args.warmup_epochs:
            warm = epoch / max(1, args.warmup_epochs)
            optimizer.param_groups[0]["lr"] = args.lr * warm
            optimizer.param_groups[1]["lr"] = args.lr_wave * warm

        # train
        unwrap(model).train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]",
                    disable=(not is_main) or args.no_pbar)
        optimizer.zero_grad(set_to_none=True)
        checksum_before = param_checksum(unwrap(model))

        for i, batch in enumerate(pbar):
            if batch is None: continue
            gt_depth = batch["gt_depth"].to(args.device, non_blocking=True)  # (B,1,H,W), meters
            rgb = batch["rgb"].to(args.device, non_blocking=True)            # (B,3,H,W)

            intr = intrinsics_base
            if train_size is not None:
                H0, W0 = gt_depth.shape[-2:]; Ht, Wt = train_size
                if (Ht, Wt) != (H0, W0):
                    gt_depth = F.interpolate(gt_depth, size=(Ht, Wt), mode='bilinear', align_corners=False)
                    rgb = F.interpolate(rgb, size=(Ht, Wt), mode='bilinear', align_corners=False)
                    sx, sy = Wt / W0, Ht / H0
                    intr = scale_intrinsics(intrinsics_base, sx, sy)

            device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
            ctx = torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)
            with ctx:
                pred_xk, pred_depth, extras = unwrap(model).forward_from_depth_train(
                    gt_depth, rgb=rgb, intrinsics=intr, to_unit=True
                )


                if is_main and i == 0:
                    print("[dbg/train] I_pred01:",
                        f"{extras['I_pred01'].min().item():.3e}",
                        f"{extras['I_pred01'].mean().item():.3e}",
                        f"{extras['I_pred01'].max().item():.3e}")
                    xk = pred_xk.detach()
                    print("[dbg/train] xk:",
                        f"{xk.min().item():.3e}",
                        f"{xk.mean().item():.3e}",
                        f"{xk.max().item():.3e}")
                    print("[dbg/train] pred_depth:",
                        f"{pred_depth.min().item():.3e}",
                        f"{pred_depth.mean().item():.3e}",
                        f"{pred_depth.max().item():.3e}")
                # ==========================================

                total, logs = loss_fn(
                    pred_depth=pred_depth, gt_depth=gt_depth, pred_xk=pred_xk,
                    obs_four_phase01=extras["I_obs01"], pred_four_phase01=extras["I_pred01"]
                )
                loss = total / args.accum

            if not _finite_or_skip(total, f"epoch {epoch} iter {i}"):
                optimizer.zero_grad(set_to_none=True); continue

            loss.backward()
            grads_finite = True
            for p in unwrap(model).parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break
            if (not torch.isfinite(total)) or (not grads_finite):
                if is_main: print("[guard] skip step due to non-finite loss/grad")
                optimizer.zero_grad(set_to_none=True)
                continue
            do_step = ((i + 1) % args.accum == 0) or ((i + 1) == len(train_loader))
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.clip_grad)
                optimizer.step()
                unwrap(model).sanitize_()
                if use_ema: ema.update(unwrap(model))
                optimizer.zero_grad(set_to_none=True)


            if is_main and i == 0 and (not args.no_pbar):
                print(f"[train stats] I_obs01 {tensor_stats(extras['I_obs01'])} | "
                      f"pred_depth {tensor_stats(pred_depth)} | gt_depth {tensor_stats(gt_depth)}")

            if is_main and (not args.no_pbar):
                pbar.set_postfix({
                    "loss": float(total.detach()),
                    "mae": float(logs.get("mae", 0.0)),
                    "ssim": float(logs.get("ssim", 0.0)),
                    "zncc": float(logs.get("zncc", 0.0)),
                    "lr": optimizer.param_groups[0]["lr"],
                })

        checksum_after = param_checksum(unwrap(model))
        if is_main:
            print(f"[epoch {epoch}] Δparam={checksum_after - checksum_before:.4e}")

        # validate (all ranks compute, reduce later)
        unwrap(model).eval()
        if use_ema: ema.apply(unwrap(model))

        val_loss_sum, val_mae_sum, val_ssim_sum, val_zncc_sum, val_n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            if is_dist:
                test_loader.sampler.set_epoch(epoch)
            pbar_v = tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]",
                          disable=(not is_main) or args.no_pbar)
            for i, batch in enumerate(pbar_v):
                if batch is None: continue
                gt_depth = batch["gt_depth"].to(args.device, non_blocking=True)
                rgb = batch["rgb"].to(args.device, non_blocking=True)
                pred_xk, pred_depth, extras = unwrap(model).forward_from_depth_train(
                    gt_depth, rgb=rgb, intrinsics=intrinsics_base, to_unit=True
                )
                if args.tta_flip:
                    gt_depth_flip = torch.flip(gt_depth, dims=[-1])
                    rgb_flip = torch.flip(rgb, dims=[-1])
                    _, pred_depth_flip, _ = unwrap(model).forward_from_depth_train(
                        gt_depth_flip, rgb=rgb_flip, intrinsics=intrinsics_base, to_unit=True
                    )
                    pred_depth = 0.5 * (pred_depth + torch.flip(pred_depth_flip, dims=[-1]))
                total, logs = loss_fn(
                    pred_depth=pred_depth, gt_depth=gt_depth, pred_xk=pred_xk,
                    obs_four_phase01=extras["I_obs01"], pred_four_phase01=extras["I_pred01"]
                )
                bs = gt_depth.size(0)
                val_loss_sum += float(total) * bs
                val_mae_sum  += float(logs.get("mae", 0.0)) * bs
                val_ssim_sum += float(logs.get("ssim", 0.0)) * bs
                val_zncc_sum += float(logs.get("zncc", 0.0)) * bs
                val_n += bs

                if is_main and i == 0:
                    print(f"[valid stats] pred_depth {tensor_stats(pred_depth)} | "
                          f"gt_depth {tensor_stats(gt_depth)}")

        # all-reduce val metrics
        t = torch.tensor([val_loss_sum, val_mae_sum, val_ssim_sum, val_zncc_sum, val_n],
                         device=args.device, dtype=torch.float64)
        if is_dist:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss = (t[0] / max(t[4], 1)).item()
        val_mae  = (t[1] / max(t[4], 1)).item()
        val_ssim = (t[2] / max(t[4], 1)).item()
        val_zncc = (t[3] / max(t[4], 1)).item()
        if is_main:
            print(f"[epoch {epoch}] val loss: {val_loss:.6f} | val mae(m): {val_mae:.6f} | "
                  f"val ssim: {val_ssim:.6f} | val zncc: {val_zncc:.6f}")

        # Save best/last only on rank0（以 val_loss 为保存 best 的标准，但会打印 best_mae）
        if is_main:
            ckpt_dir = Path(args.save_dir) / "checkpoints"
            improved_any = False
            if val_loss < best_val:
                best_val = val_loss; improved_any = True
            if val_mae  < best_mae:
                best_mae = val_mae; improved_any = True
            if improved_any:
                torch.save({"model": unwrap(model).state_dict(), "epoch": epoch, "val_loss": val_loss, "val_mae": val_mae}, ckpt_dir / "best.pt")
                print(f"  -> saved best (val_loss {val_loss:.6f}, val_mae {val_mae:.6f})")
            torch.save({"model": unwrap(model).state_dict(), "epoch": epoch, "val_loss": val_loss, "val_mae": val_mae}, ckpt_dir / "last.pt")


        scheduler.step(val_loss)


        stop_flag = torch.tensor([0], device=args.device)
        if args.early_stop and is_main:
            es_metric_val = val_mae if args.es_metric == "val_mae" else val_loss
            improved, should_stop, best, no_improve = es.update(epoch, es_metric_val)
            if is_main:
                print(f"[early-stop] metric={args.es_metric} cur={es_metric_val:.6f} best={best:.6f} "
                      f"no_improve={no_improve}/{args.es_patience} (min_delta={args.es_min_delta})")
            if should_stop and epoch >= max(args.es_warmup_epochs, 1):
                print(f"[early-stop] triggered at epoch {epoch}.")
                stop_flag[0] = 1

        if is_dist:
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            should_break_all = True

        if should_break_all:
            break

    # -------- Final export (rank0 only) --------
    if is_main:
        print("[final export] start ...")
        out_dir = Path(args.save_dir) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)


        best_path = Path(args.save_dir) / "checkpoints" / "best.pt"
        if best_path.exists():
            print(f"[final export] loading best from: {best_path}")
            state = torch.load(best_path, map_location=args.device)
            unwrap(model).load_state_dict(state["model"], strict=False)
        else:
            print("[final export] WARN: best.pt not found; export with current weights.")
        unwrap(model).eval()
        if use_ema: ema.apply(unwrap(model))

        all_idx = list_indices(args.gt_dir)

        maes, rmses, counted = [], [], 0

        with torch.no_grad():
            last_xk = None
            for n in tqdm(all_idx, desc="[final export]"):
                gt_path = Path(args.gt_dir) / f"depth_{n:04d}.png"
                rgb_path = Path(args.rgb_dir) / args.rgb_tmpl.format(n)
                if not gt_path.exists(): continue

                gt_depth = read_depth_image(str(gt_path), args.depth_scale).unsqueeze(0).unsqueeze(0).to(args.device)  # meters

                if not rgb_path.exists():
                    alt = [rgb_path.with_suffix(".jpg"), rgb_path.with_suffix(".jpeg")]
                    rgb_path = next((p for p in alt if p.exists()), None)
                if rgb_path is None: 

                    continue
                rgb = read_rgb_image(str(rgb_path)).unsqueeze(0).to(args.device)

                pred_xk, pred_depth, _ = unwrap(model).forward_from_depth_train(
                    gt_depth, rgb=rgb, intrinsics=intrinsics_base, to_unit=True
                )
                last_xk = pred_xk

                # 导出：16-bit 显示编码 + 伪彩
                save_depth_16bit_and_rgb(out_dir, pred_depth[0,0], n, max_depth_vis=args.dmax)
                # 导出：毫米编码（与 GT 完全同编码）
                save_depth_16bit_mm(out_dir,        pred_depth[0,0], n)

                # 评估：米制 MAE/RMSE（忽略无效 GT）
                valid = (gt_depth > 0) & torch.isfinite(gt_depth) & torch.isfinite(pred_depth)
                if valid.any():
                    diff = (pred_depth - gt_depth)[valid]
                    mae  = torch.mean(torch.abs(diff)).item()
                    rmse = torch.sqrt(torch.mean(diff**2)).item()
                    maes.append(mae); rmses.append(rmse); counted += 1

            if last_xk is not None:
                save_xk_global(out_dir, unwrap(model).corr.freqs_hz, last_xk)


        metrics = {}
        if counted > 0:
            metrics["MAE_m"]  = float(np.mean(maes))
            metrics["RMSE_m"] = float(np.mean(rmses))
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[final export] saved to: {str(out_dir)}")
        if counted > 0:
            print(f"[final export] metrics: MAE={metrics['MAE_m']:.6f} m, RMSE={metrics['RMSE_m']:.6f} m")
        print("Done.")

    cleanup_distributed_mode()


if __name__ == "__main__":
    main()
