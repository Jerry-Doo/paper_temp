# inference.py
import argparse, os, re
from pathlib import Path
import numpy as np
from PIL import Image

import torch

# 使用你的工程命名
from modelone import PhaseMambaNet, depth_to_phase_t, build_freq_list

def read_gray(path: Path) -> np.ndarray:
    # 兼容 8/16bit 灰度
    # PNG/TIFF 用 "I" 模式读 16bit，其它回落到 L（8bit）
    try:
        im = Image.open(path)
        if im.mode in ("I;16", "I;16L", "I"):
            im = im.convert("I")
        else:
            im = im.convert("L")
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {path}\n{e}")
    arr = np.array(im)
    return arr.astype(np.float32)

def to01(arr: np.ndarray, div: float = None) -> np.ndarray:
    if div is not None and div > 0:
        return np.clip(arr / float(div), 0.0, 1.0)
    vmax = float(arr.max())
    if vmax <= 1.5:
        return np.clip(arr, 0.0, 1.0)
    # 粗略猜位数
    if vmax <= 255.0: d = 255.0
    elif vmax <= 4095.0: d = 4095.0
    else: d = 65535.0
    return np.clip(arr / d, 0.0, 1.0)

def find_four_in_dir(d: Path):
    d = Path(d)
    cands = [p for p in d.iterdir() if p.is_file()]
    if len(cands) < 4:
        raise FileNotFoundError(f"Not enough files in {d}")
    # 优先匹配 0/1/2/3
    slots = {0: None, 1: None, 2: None, 3: None}
    for p in cands:
        m = re.search(r'(\d+)', p.stem)
        if m:
            idx = int(m.group(1))
            if idx in slots and slots[idx] is None:
                slots[idx] = p
    if all(slots[k] is not None for k in range(4)):
        return [slots[k] for k in range(4)]
    # 退化为字母序前 4 张
    return sorted(cands)[:4]

def save_depth_16bit_and_rgb(out_dir: Path, depth_m: torch.Tensor, index: int, max_depth_vis: float = 10.0):
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

def autodetect_dim(state_dict: dict, fallback: int = 128) -> int:
    # 依据 embed conv 的 out_channels 推断 dim
    for k, v in state_dict.items():
        if k.endswith("backbone.embed.0.weight"):  # (dim, in_ch, 7, 7)
            return v.shape[0]
    return fallback

@torch.no_grad()
def run_infer(args):
    device = args.device
    # ---- 载入权重并构建网络 ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    dim = args.dim or autodetect_dim(state, fallback=128)
    from modelone import PhaseMambaNet
    K = int(build_freq_list(args.base_freq_hz).numel())
    # xk_min/xk_max 只是占位，真正的值由 checkpoint 覆盖
    xk_min = torch.zeros(2*K); xk_max = torch.ones(2*K)

    model = PhaseMambaNet(
        dim=dim, num_blocks=args.blocks, in_ch=4,
        base_freq_hz=args.base_freq_hz, dmax=args.dmax,
        corr_norm_mode=args.corr_norm_mode, corr_dc_baseline=args.corr_dc,
        use_global_xk=True, xk_param_mode="post_box",
        xk_min=xk_min, xk_max=xk_max
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[warn] missing keys: {len(missing)}")
    if unexpected: print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval().to(device)

    # ---- 读入 4 相位图 ----
    if args.four_dir:
        paths = find_four_in_dir(Path(args.four_dir))
    else:
        assert args.i0 and args.i1 and args.i2 and args.i3, "Provide --four_dir or all of --i0..--i3"
        paths = [Path(args.i0), Path(args.i1), Path(args.i2), Path(args.i3)]
    imgs = [to01(read_gray(p), args.div) for p in paths]
    H, W = imgs[0].shape
    arr = np.stack(imgs, axis=0)  # (4,H,W)
    ten = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,4,H,W), 0..1

    # ---- 推理链路：I → backbone → d1 → Synth(d1) → R → Refine → d2 ----
    I = ten.clamp(0, 1)
    feat, depth_raw, _ = model.backbone(I)
    d1 = (torch.tanh(depth_raw) * 0.5 + 0.5) * model.dmax
    d1 = torch.nan_to_num(d1, nan=0.0, posinf=model.dmax, neginf=0.0).clamp(0, model.dmax)

    xk = model.get_global_xk().to(device)
    xkB = xk.expand(d1.size(0), -1)

    t_map_pred1 = depth_to_phase_t(d1)
    pred4_env1 = model.corr.sample_map_env(
        xkB, t_map_pred1,
        beta=model.env_beta, kappa=model.env_kappa,
        lam_d=model.env_lam_d, gain=model.env_gain,
        alpha=1.0, use_falloff=args.use_falloff, for_zncc=True)
    pred4_01_1 = model.corr.to_unit(pred4_env1)
    R = (I - pred4_01_1).clamp(-1, 1)

    r_feat = model.res_embed(R)
    d_feat = model.dep_embed(d1)
    rf = model.refine_fuse(torch.cat([feat, r_feat, d_feat], dim=1))
    delta_raw = model.delta_head(rf)
    d2 = (d1 + 0.1 * model.dmax * torch.tanh(delta_raw)).clamp(0, model.dmax)

    # 可选：用 d2 再合成 4 相位（调试/可视化）
    if args.save_phase:
        t_map_pred2 = depth_to_phase_t(d2)
        pred4_env2 = model.corr.sample_map_env(
            xkB, t_map_pred2,
            beta=model.env_beta, kappa=model.env_kappa,
            lam_d=model.env_lam_d, gain=model.env_gain,
            alpha=1.0, use_falloff=args.use_falloff, for_zncc=True)
        pred4_01_2 = model.corr.to_unit(pred4_env2)
    else:
        pred4_01_2 = None

    # ---- 保存结果 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_depth_16bit_and_rgb(out_dir, d2[0,0], 0, max_depth_vis=args.max_depth_vis)
    if pred4_01_2 is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for k in range(4):
            img = (pred4_01_2[0,k].detach().cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"pred_phase{k}_0000.png")

    print("Saved to:", out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint (.pt)")
    ap.add_argument("--i0", type=str); ap.add_argument("--i1", type=str)
    ap.add_argument("--i2", type=str); ap.add_argument("--i3", type=str)
    ap.add_argument("--four_dir", type=str, help="directory containing 4 phase images")
    ap.add_argument("--div", type=float, default=None, help="divide raw values to scale into [0,1] (e.g., 255 or 4095)")
    ap.add_argument("--out_dir", type=str, default="outputs_infer")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # must match training
    ap.add_argument("--base_freq_hz", type=float, default=50e6)
    ap.add_argument("--dmax", type=float, default=10.0)
    ap.add_argument("--dim", type=int, default=None, help="auto-detect if None")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--corr_norm_mode", type=str, default="fixmean", choices=["none","fixmean","fixl2"])
    ap.add_argument("--corr_dc", type=float, default=1.0)
    ap.add_argument("--use_falloff", dest="use_falloff", action="store_true")
    ap.add_argument("--no_falloff", dest="use_falloff", action="store_false")
    ap.set_defaults(use_falloff=True)
    ap.add_argument("--max_depth_vis", type=float, default=10.0)
    ap.add_argument("--save_phase", action="store_true", help="also save synthesized 4-phase from d2")
    args = ap.parse_args()
    run_infer(args)

if __name__ == "__main__":
    main()
