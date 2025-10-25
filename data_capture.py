import os
import numpy as np
import imageio.v2 as imageio

input_dir = 'nyu_output_rawdump'
output_root = 'nyu_output_fixed'
os.makedirs(output_root, exist_ok=True)

targets = {
    'depths.npy': 'depths',
    'images.npy': 'images',
    'labels.npy': 'labels',
    'rawDepths.npy': 'raw_depths'
}

def hw_fix(x):
    """如果高宽颠倒(如 640x480)，交换到 480x640；支持 2D/3D"""
    if x.ndim == 2 and x.shape[0] > x.shape[1]:
        return x.T
    if x.ndim == 3 and x.shape[0] > x.shape[1] and x.shape[2] in (1,3,4):
        return np.transpose(x, (1,0,2))
    return x

for npy_file, folder_name in targets.items():
    arr = np.load(os.path.join(input_dir, npy_file))
    out_dir = os.path.join(output_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # NCHW -> NHWC（RGB/RGBA）
    if arr.ndim == 4 and arr.shape[1] in [3, 4]:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))  # 单张彩色

    for i in range(arr.shape[0]):
        img = hw_fix(arr[i])

        # ✅ 针对 RGB 图像才做归一化
        if 'images' in folder_name:
            if np.issubdtype(img.dtype, np.floating):
                mn, mx = float(np.min(img)), float(np.max(img))
                if mx > mn:
                    img = (img - mn) / (mx - mn)
                img = (img * 255).clip(0, 255).astype(np.uint8)

        # ✅ 深度图保持原格式
        elif 'depth' in folder_name:
            if np.issubdtype(img.dtype, np.floating):
                # 保存为 uint16 毫米深度
                img = (img * 1000).astype(np.uint16)

        # squeeze 单通道
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]

        ext = '.png'
        out_path = os.path.join(out_dir, f"{folder_name[:-1]}_{i:04d}{ext}")
        imageio.imwrite(out_path, img)

    print(f"✅ {npy_file} 已导出至 {out_dir}")
