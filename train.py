# train.py — Physics-aware iToF trainer
# -------------------------------------------------
# Use GT depth to synthesize 4-phase in-graph, with differentiable physics.
#
# Example:
#   python train.py --gt_dir data/depths --save_dir runs/paper \
#     --export_four_phase --zncc_w 0.25 --dim 128 --blocks 8 --amp --ema
#   # 可选排障：--no_noise  或  --freeze_wave_epochs 3

import os, re, argparse, random, json, shutil
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from modelone import PhaseMambaNet, PhaseMambaPhysLoss

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
        if depth_scale is not None and depth.max() > 1.5:
            depth = depth / float(depth_scale)
    return depth

# --------------- Dataset (depth only) ---------------
class DepthOnlyDataset(Dataset):
    def __init__(self, gt_dir: Union[str, Path], indices: List[int], depth_scale: Optional[float] = 1000.0):
        self.gt_dir = Path(gt_dir)
        self.indices = indices
        self.depth_scale = depth_scale
        self.gt_tmpl = "depth_{:04d}.png"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        n = self.indices[i]
        gt_path = self.gt_dir / self.gt_tmpl.format(n)
        gt_depth = read_depth_image(str(gt_path), self.depth_scale)  # (H,W)
        return {
            "gt_depth": gt_depth.float().unsqueeze(0),   # (1,H,W)
            "index": n
        }

# --------------- split helpers ---------------
def build_splits_depth(gt_dir: Union[str, Path]) -> Tuple[List[int], List[int]]:
    gt_dir = Path(gt_dir)
    pat = re.compile(r"depth_(\d+)\.png")
    idxs = []
    for fn in os.listdir(gt_dir):
        m = pat.match(fn)
        if m:
            idxs.append(int(m.group(1)))
    idxs_s = sorted(idxs)
    n = len(idxs_s)
    k = int(n * 0.7) if n >= 10 else max(1, n * 7 // 10)
    train_idx = idxs_s[:k]
    test_idx = idxs_s[k:]
    return train_idx, test_idx

# --------------- utils ---------------
def param_checksum(model) -> float:
    with torch.no_grad():
        s = 0.0
        for p in model.parameters():
            if p.requires_grad:
                s += float(p.detach().abs().sum())
        return s

def tensor_stats(x: torch.Tensor) -> str:
    x = x.detach().float()
    return f"min={float(x.min()):.4f} mean={float(x.mean()):.4f} max={float(x.max()):.4f}"

def save_signal_expression(expr_dir: Path, freqs_hz: torch.Tensor, xk: torch.Tensor,
                           index: int, dc_baseline: float = 1.0):
    expr_dir.mkdir(parents=True, exist_ok=True)
    freqs = freqs_hz.detach().cpu().numpy().tolist()
    xk = xk.detach().cpu().squeeze().float()
    K = len(freqs)
    if xk.numel() == 2 * K:
        a = xk[:K]; s = xk[K:]
    elif xk.numel() == K:
        a = xk; s = torch.zeros_like(a)
    else:
        raise ValueError(f"xk length {xk.numel()} not in {{K,2K}} where K={K}")
    l1 = (a.abs() + s.abs()).sum().clamp(min=1e-12)
    scale = dc_baseline / l1
    a_n = (a * scale).numpy().tolist()
    s_n = (s * scale).numpy().tolist()
    b = float(dc_baseline)

    expr = [
        f"y(t) = {b:.6f} + Σ_k [ a_k cos(2π f_k t) + s_k sin(2π f_k t) ]",
        f"Constraints: Σ_k(|a_k|+|s_k|) = {b:.6f}  ->  y(t) ∈ [0, {2*b:.6f}]",
        "where:"
    ]
    for i, f in enumerate(freqs):
        expr.append(f"  a_{i+1} = {a_n[i]:+.6f}, s_{i+1} = {s_n[i]:+.6f}, f_{i+1} = {f:.3e} Hz")

    (expr_dir / f"expr_{index:04d}.txt").write_text("\n".join(expr), encoding="utf-8")
    with open(expr_dir / f"expr_{index:04d}.json", "w") as fp:
        json.dump({
            "formula": "y(t)=b+Σ[a_k cos(2π f_k t)+s_k sin(2π f_k t)]",
            "a_coeffs_l1norm": a_n,
            "s_coeffs_l1norm": s_n,
            "freqs_hz": freqs,
            "dc_baseline": b,
            "note": "a/s have been L1-normalized so that Σ(|a|+|s|)=b; waveform in [0, 2*b]"
        }, fp, ensure_ascii=False, indent=2)

def _sanitize_img(x: torch.Tensor, peak: float) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=peak, neginf=0.0)
    return x.clamp(0, peak) / peak

def save_depth_16bit_and_rgb(out_dir: Path, depth_m: torch.Tensor, index: int, max_depth_vis: float = 3.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = depth_m.detach().cpu().float().clamp(0, max_depth_vis)   # (H,W)
    d16 = (d / max_depth_vis * 65535.0).round().clamp(0, 65535).to(torch.uint16).numpy()
    Image.fromarray(d16).save(out_dir / f"depth16_{index:04d}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("turbo")
    dn = (d / max_depth_vis).numpy()
    rgb = (cm(np.clip(dn, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_dir / f"depth16_{index:04d}_rgb.png")

def save_four_phase_processed(out_dir: Path, phase4: torch.Tensor, index: int,
                              prefix: str = "synth_phase", peak: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _sanitize_img(phase4.detach().cpu().float(), peak)
    for k in range(4):
        arr = (p[k].numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{prefix}{k}_{index:04d}.png")

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# --------------- EMA ---------------
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
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: torch.nn.Module):
        if self.backup is not None:
            model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
            self.backup = None

# --------------- main ---------------
def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder with GT depth PNGs depth_XXXX.png")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Divide raw depth by this if uint16/uint8")
    # Physics options
    parser.add_argument("--use_falloff", action="store_true", help="Enable geometric falloff (d0/z)^2 in synthesis")
    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4, help="LR for backbone/depth/env")
    parser.add_argument("--lr_wave", type=float, default=2e-5, help="LR for global waveform (smaller)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use AMP (bfloat16)")
    parser.add_argument("--max_depth_vis", type=float, default=10.0)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--dmax", type=float, default=10.0, help="Depth range upper bound for head mapping")
    parser.add_argument("--base_freq_hz", type=float, default=50e6)
    # Scheduler
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--rewind_on_plateau", action="store_true")
    # Tricks
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--no_pbar", action="store_true")
    # Export/vis
    parser.add_argument("--export_four_phase", action="store_true")
    parser.add_argument("--keep_all_epochs", action="store_true")
    parser.add_argument("--show_four_phase_log", action="store_true")
    # Physics/regularization
    parser.add_argument("--edge_w", type=float, default=0.0)
    parser.add_argument("--corr_dc", type=float, default=1.0, help="b: DC baseline (max=2*b)")
    parser.add_argument("--corr_norm_mode", type=str, default="none", choices=["none","fixmean","fixl2"])
    parser.add_argument("--freq_l1_w", type=float, default=0.0, help="frequency-weighted L1 prior (0=off)")
    parser.add_argument("--freq_alpha", type=float, default=1.0)
    # 可选项：ZNCC 权重（from-depth 推荐 0.25）
    parser.add_argument("--zncc_w", type=float, default=0.25)
    # 可选：早期冻结波形，避免发散
    parser.add_argument("--freeze_wave_epochs", type=int, default=0)
    # 可选：禁用合成噪声，用于排障
    parser.add_argument("--no_noise", action="store_true")
    # 兼容旗标（忽略）
    parser.add_argument("--train_from_depth", action="store_true",
                        help="(ignored) From-depth is the only mode.")

    args = parser.parse_args()
    if args.train_from_depth:
        print("[Note] --train_from_depth is ignored (from-depth is the only mode).")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "vis").mkdir(parents=True, exist_ok=True)
    (save_dir / "expr").mkdir(parents=True, exist_ok=True)

    # ---------- Splits & Datasets ----------
    train_idx, test_idx = build_splits_depth(args.gt_dir)
    print(f"[Mode: from_depth] Train items: {len(train_idx)}, Test items: {len(test_idx)}")
    train_ds = DepthOnlyDataset(args.gt_dir, train_idx, depth_scale=args.depth_scale)
    test_ds  = DepthOnlyDataset(args.gt_dir, test_idx,  depth_scale=args.depth_scale)

    pin = args.device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)

    # ---------- x_k bounds (K=15 -> 2K=30) ----------
    cos_min = torch.tensor([
        0.0247367180319292, -0.218511647188518, -0.128039900641918, -0.188089030029964,
        -0.101271446313554, -0.0739265690272022, -0.0517185888075761, -0.0470680503186579,
        -0.0429835203719536, -0.0113244401026370, -0.0867256924442522, -0.0398804608758447,
        -0.0255234949605406, -0.0204187959684325, -0.0163350367747460
    ], dtype=torch.float32)
    cos_max = torch.tensor([
        1.000000562151060, 0.246269657347430, 0.172776779981168, 0.067885132423475,
        0.054348930049410, 0.0110355732644022, 0.0079027851216088, 0.0073635999741769,
        0.0075220109866885, 0.0071864386286823, 0.0059862393148076, 0.0108529551727735,
        0.00507913000000000, 0.0050842000000000, 0.0044896000000000
    ], dtype=torch.float32)
    sin_min = torch.tensor([
        -0.292903812153803, -0.133624174055044, -0.0765348572830997, -0.0384905669003143,
        -0.0313234820028485, -0.0303887103998798, -0.0452395377084766, -0.0344357506637897,
        -0.0309456571202258, -0.0212468035562158, -0.0146363445052302, -0.0149122298097099,
        -0.0127050000000000, -0.0108800000000000, -0.0094200000000000
    ], dtype=torch.float32)
    sin_max = torch.tensor([
        0.305261167400373, 0.119887020570952, 0.0706313176834897, 0.0397174345064721,
        0.0344136176416263, 0.0266434058843329, 0.0403971596620880, 0.00367550978888006,
        0.00348480445863076, 0.00284442773359534, 0.00325240839414840, 0.00268676948278312,
        0.00241000000000000, 0.00213000000000000, 0.00189000000000000
    ], dtype=torch.float32)
    xk_min = torch.cat([cos_min, sin_min], dim=0)  # [30]
    xk_max = torch.cat([cos_max, sin_max], dim=0)  # [30]

    # ---------- Model ----------
    model = PhaseMambaNet(
        dim=args.dim, num_blocks=args.blocks,
        in_ch=4,
        use_corr_synth=True,
        base_freq_hz=args.base_freq_hz,
        dmax=args.dmax,
        corr_norm_mode=args.corr_norm_mode,
        corr_dc_baseline=args.corr_dc,
        corr_use_soft_clip=True,
        use_global_xk=True,
        xk_param_mode="post_box",
        xk_min=xk_min, xk_max=xk_max,
    ).to(args.device)

    # 关闭噪声（排障选项）
    if args.no_noise:
        model._noise_read = (0.0, 0.0)
        model._noise_shot = (0.0, 0.0)
        model._noise_gain = (1.0, 1.0)
        model._noise_offs = (0.0, 0.0)
        model._noise_fpnr = (0.0, 0.0)
        model._noise_fpnc = (0.0, 0.0)
        print("[Info] Paper-like sensor noise is DISABLED (--no_noise).")

    # ---------- Loss ----------
    loss_fn = PhaseMambaPhysLoss(
        mae_w=1.0, ssim_w=0.5, zncc_w=args.zncc_w,
        edge_w=args.edge_w,
        ssim_use_depth_unit=True,
        ssim_max_val=1.0,
        dmax_for_ssim=args.max_depth_vis,
        base_freq_hz=args.base_freq_hz,
        corr_norm_mode=args.corr_norm_mode,
        corr_dc_baseline=args.corr_dc,
        corr_use_soft_clip=True,
        freq_l1_w=args.freq_l1_w,
        freq_alpha=args.freq_alpha,
        zncc_to_depth=True
    ).to(args.device)

    # ---------- Optimizer: backbone/env vs waveform ----------
    base_params, wave_params = [], []
    for n, p in model.named_parameters():
        if n == "global_xk_logits":     # post_box mode
            wave_params.append(p)
        else:
            base_params.append(p)
    print("wave params size:", sum(p.numel() for p in wave_params))

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": wave_params, "lr": args.lr_wave, "weight_decay": 0.0},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                  patience=args.patience, min_lr=args.lr_min, verbose=True)

    use_amp = args.amp and args.device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    best_val = float("inf")
    best_ckpt = save_dir / "checkpoints" / "best.pt"

    # EMA / Warmup
    use_ema = args.ema
    ema = EMA(model, decay=args.ema_decay) if use_ema else None
    base_lr = args.lr
    wave_lr = args.lr_wave

    def _finite_or_skip(total_loss, batch_dbg: str) -> bool:
        if not torch.isfinite(total_loss):
            print(f"[skip batch] non-finite loss at {batch_dbg}")
            return False
        return True

    for epoch in range(1, args.epochs + 1):
        # 冻结波形参数（可选）
        freeze_wave = epoch <= max(0, args.freeze_wave_epochs)
        for g in optimizer.param_groups:
            if "params" in g and len(g["params"]) > 0:
                pass
        if freeze_wave and len(wave_params) > 0:
            for p in wave_params: p.requires_grad_(False)
        else:
            for p in wave_params: p.requires_grad_(True)

        # -------- Train --------
        model.train()
        if epoch <= args.warmup_epochs:
            warm = epoch / max(1, args.warmup_epochs)
            optimizer.param_groups[0]["lr"] = base_lr * warm
            optimizer.param_groups[1]["lr"] = wave_lr * warm

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", disable=args.no_pbar)
        optimizer.zero_grad(set_to_none=True)
        checksum_before = param_checksum(model)

        for i, batch in enumerate(pbar):
            gt_depth = batch["gt_depth"].to(args.device, non_blocking=True)
            device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
            ctx = torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)
            with ctx:
                pred_xk, pred_depth, extras = model.forward_from_depth_train(
                    gt_depth, to_unit=True, use_falloff=args.use_falloff
                )
                total, logs = loss_fn(
                    pred_depth=pred_depth,
                    gt_depth=gt_depth,
                    pred_xk=pred_xk,
                    obs_four_phase=extras.get("I_obs_paper_like01"),
                    pred_four_phase_pred01=extras.get("pred_four_phase_env01")
                )
                loss = total / args.accum

            if not _finite_or_skip(total, f"epoch {epoch} iter {i}"):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            do_step = ((i + 1) % args.accum == 0) or ((i + 1) == len(train_loader))
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                # >>> 关键：step 后立刻清洗+约束，避免 NaN 继续扩散
                if hasattr(model, "sanitize_"):
                    model.sanitize_()
                elif hasattr(model, "clamp_env_"):
                    model.clamp_env_()
                if use_ema:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            if not args.no_pbar:
                pbar.set_postfix({
                    "loss": float(total.detach()),
                    "mae":  float(logs.get("mae", 0.0)),
                    "ssim": float(logs.get("ssim", 0.0)),
                    "zncc": float(logs.get("zncc", 0.0)),
                    "lr":   optimizer.param_groups[0]["lr"],
                    "lr_w": optimizer.param_groups[1]["lr"],
                })

        checksum_after = param_checksum(model)
        print(f"[epoch {epoch}] param_delta={(checksum_after - checksum_before):.4e}  "
              f"env(gain={float(model.env_gain.detach()):.3f}, beta={float(model.env_beta.detach()):.3f}, "
              f"lam_d={float(model.env_lam_d.detach()):.3f})")

        # -------- Validate --------
        model.eval()
        if use_ema:
            ema.apply(model)

        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            pbar_v = tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]",
                          disable=(args.no_pbar or args.export_four_phase))

            # Output dirs (per-epoch or rolling)
            if args.keep_all_epochs:
                vis_ep_dir  = (save_dir / "vis"  / f"ep_{epoch:03d}")
                expr_ep_dir = (save_dir / "expr" / f"ep_{epoch:03d}")
                vis_ep_dir.mkdir(parents=True, exist_ok=True)
                expr_ep_dir.mkdir(parents=True, exist_ok=True)
            else:
                vis_ep_dir  = (save_dir / "vis")
                expr_ep_dir = (save_dir / "expr")
                reset_dir(vis_ep_dir)
                reset_dir(expr_ep_dir)

            for i, batch in enumerate(pbar_v):
                gt_depth = batch["gt_depth"].to(args.device, non_blocking=True)
                pred_xk, pred_depth, extras = model.forward_from_depth_train(
                    gt_depth, to_unit=True, use_falloff=args.use_falloff
                )
                if args.tta_flip:
                    gt_depth_flip = torch.flip(gt_depth, dims=[-1])
                    _, pred_depth_flip, _ = model.forward_from_depth_train(
                        gt_depth_flip, to_unit=True, use_falloff=args.use_falloff
                    )
                    pred_depth = 0.5 * (pred_depth + torch.flip(pred_depth_flip, dims=[-1]))

                total, logs = loss_fn(
                    pred_depth=pred_depth,
                    gt_depth=gt_depth,
                    pred_xk=pred_xk,
                    obs_four_phase=extras.get("I_obs_paper_like01"),
                    pred_four_phase_pred01=extras.get("pred_four_phase_env01")
                )

                bs = gt_depth.size(0)
                val_loss += float(total) * bs
                val_n    += bs

                if i == 0:
                    print("[valid stats]",
                          "pred_depth", tensor_stats(pred_depth),
                          "| gt_depth", tensor_stats(gt_depth))

                if args.export_four_phase:
                    for b in range(bs):
                        idx = int(batch["index"][b])
                        I_obs = extras.get("I_obs_paper_like01")[b]
                        save_four_phase_processed(vis_ep_dir, I_obs, idx, prefix="obs_paper01", peak=1.0)
                        #synth4 = extras.get("pred_four_phase_env")[b] if "pred_four_phase_env" in extras else None
                        #if synth4 is not None:
                        #    save_four_phase_processed(vis_ep_dir, synth4, idx, prefix="pred_phase", peak=2*args.corr_dc)
                        save_depth_16bit_and_rgb(vis_ep_dir, pred_depth[b, 0], idx,
                                                 max_depth_vis=args.max_depth_vis)
                        save_signal_expression(expr_ep_dir, model.corr.freqs_hz, pred_xk[b].cpu(), idx,
                                               dc_baseline=args.corr_dc)

        val_loss = val_loss / max(val_n, 1)
        print(f"[epoch {epoch}] val loss: {val_loss:.6f}")

        if use_ema:
            ema.restore(model)

        # Save best/last
        ckpt_dir = save_dir / "checkpoints"
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val}, best_ckpt)
            print(f"  -> saved best to {best_ckpt} (val {best_val:.6f})")

        last_path = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss}, last_path)

        # LR schedule (+ optional rewind)
        prev_lr0 = optimizer.param_groups[0]["lr"]
        prev_lr1 = optimizer.param_groups[1]["lr"]
        scheduler.step(val_loss)
        new_lr0 = optimizer.param_groups[0]["lr"]
        new_lr1 = optimizer.param_groups[1]["lr"]
        if args.rewind_on_plateau and (new_lr0 < prev_lr0 - 1e-12 or new_lr1 < prev_lr1 - 1e-12):
            if best_ckpt.exists():
                state = torch.load(best_ckpt, map_location=args.device)
                model.load_state_dict(state["model"])
                print(f"[plateau] LRs reduced {prev_lr0:.2e}->{new_lr0:.2e} (base), {prev_lr1:.2e}->{new_lr1:.2e} (wave); rewind to best (epoch {state.get('epoch','?')})")
        # 额外安全：每轮结束再清洗一次
        if hasattr(model, "sanitize_"):
            model.sanitize_()

    print("Done.",
          "Depth outputs ->", str(save_dir / "vis"),
          "| Expressions ->", str(save_dir / "expr"))

if __name__ == "__main__":
    main()
