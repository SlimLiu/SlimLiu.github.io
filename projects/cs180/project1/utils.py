# utils.py
import numpy as np

# ---------------- 基础工具 ----------------

def to_float01(img):
    """标准化到 float32 且落在 [0,1]。"""
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0)
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0)
    img = img.astype(np.float32)
    vmax = img.max()
    if vmax > 1.0:
        img = img / vmax
    img = np.clip(img, 0.0, 1.0)
    return img

def split(im):
    """将竖向堆叠的三通道灰度图按高度三等分为 B, G, R。
    期望 im.shape 为 (3H, W) 或 (3H, W, C=1)。
    """
    H_total = im.shape[0]
    H = H_total // 3
    # 防御：确保至少能切出 3 段
    if H * 3 > H_total:
        H = int(np.floor(H_total / 3.0))
    b = im[:H, ...]
    g = im[H:2*H, ...]
    r = im[2*H:3*H, ...]
    # 若读入是 (H, W, 3) 的彩色，请先转灰度再 split；此函数默认输入是单通道堆叠图
    return b, g, r

def roll2(img, dy, dx):
    """沿 (row, col) 做环绕平移，不改变通道维。"""
    out = np.roll(img, dy, axis=0)
    out = np.roll(out, dx, axis=1)
    return out

def crop_border(im, crop_frac=0.1):
    """裁掉四周一定比例，只保留中心区域用于匹配评分。
    支持 2D (H,W) 和 3D (H,W,C)。crop_frac ∈ [0, 0.45] 较合理。
    """
    if crop_frac <= 0:
        return im
    h, w = im.shape[:2]
    ch = int(h * crop_frac)
    cw = int(w * crop_frac)
    # 防御：避免裁没
    ch = min(ch, h // 2 - 1) if h >= 4 else 0
    cw = min(cw, w // 2 - 1) if w >= 4 else 0
    if ch <= 0 and cw <= 0:
        return im
    if im.ndim == 2:
        return im[ch:h - ch, cw:w - cw]
    elif im.ndim == 3:
        return im[ch:h - ch, cw:w - cw, :]
    else:
        raise ValueError(f"Unsupported ndim: {im.ndim}")

def to_gray(x):
    """将 (H,W,3) 转为灰度 (H,W)；若已是 (H,W) 则原样返回。"""
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[2] == 3:
        # ITU-R BT.601 加权
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    if x.ndim == 3 and x.shape[2] == 1:
        return x[..., 0]
    raise ValueError(f"to_gray expects (H,W) or (H,W,3)/(H,W,1), got shape {x.shape}")

# ---------------- 打分函数 ----------------

def ssd_score(a, b):
    """Sum of Squared Differences，越小越相似。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = a - b
    return float(np.sum(diff * diff))

def ncc_score(a, b):
    """Normalized Cross-Correlation（≈余弦相似度），[-1,1]，越大越相似。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a0 = a - a.mean()
    b0 = b - b.mean()
    an = np.linalg.norm(a0.ravel())
    bn = np.linalg.norm(b0.ravel())
    if an == 0 or bn == 0:
        return -np.inf
    return float(np.sum(a0 * b0) / (an * bn))

# ---------------- 核心对齐 ----------------

def align(moving, reference, search_radius=15, method='ncc', crop_frac=0.1, return_shift=False):
    """将 moving 图对齐到 reference。
    - search_radius: 在 [-R, R] 的窗口内暴力搜索 (dy, dx)
    - method: 'ncc'（取最大）或 'ssd'（取最小）
    - crop_frac: 评分时仅比较中心区域比例（提升鲁棒性）
    - return_shift: 是否返回 (aligned, (dy,dx), best_score)
    """
    # 统一到 float32 [0,1]
    moving = to_float01(moving)
    reference = to_float01(reference)

    ref_gray = to_gray(reference)
    H, W = ref_gray.shape
    # 评分只用中心区域
    ref_crop = crop_border(ref_gray, crop_frac)

    if method.lower() == 'ncc':
        score_fn = ncc_score
        better = lambda s, best: (best is None) or (s > best)
    elif method.lower() == 'ssd':
        score_fn = ssd_score
        better = lambda s, best: (best is None) or (s < best)
    else:
        raise ValueError("method must be 'ncc' or 'ssd'")

    best_score = None
    best_shift = (0, 0)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            moved = roll2(moving, dy, dx)
            a = to_gray(moved)
            a = crop_border(a, crop_frac)
            s = score_fn(a, ref_crop)
            if better(s, best_score):
                best_score = s
                best_shift = (dy, dx)

    aligned = roll2(moving, best_shift[0], best_shift[1])

    if return_shift:
        return aligned, best_shift, best_score
    return aligned





# ---------------- 下采样 & 金字塔 ----------------

def downsample2(im):
    """2x2 box-filter downsample: low-pass (mean of 4) then decimate by 2."""
    im = to_float01(im)
    if im.ndim == 2:
        H, W = im.shape
        H2, W2 = H - (H % 2), W - (W % 2)
        im = im[:H2, :W2]
        return 0.25 * (im[0::2, 0::2] + im[1::2, 0::2] + im[0::2, 1::2] + im[1::2, 1::2])
    elif im.ndim == 3:
        H, W, C = im.shape
        H2, W2 = H - (H % 2), W - (W % 2)
        im = im[:H2, :W2, :]
        return 0.25 * (im[0::2, 0::2, :] + im[1::2, 0::2, :] + im[0::2, 1::2, :] + im[1::2, 1::2, :])
    else:
        raise ValueError(f"Unsupported ndim: {im.ndim}")

def build_pyramid(im, min_size=64, max_levels=None):
    """
    Build a grayscale pyramid from coarse->fine.
    Stops when min(H,W)//2 < min_size or max_levels reached.
    """
    cur = to_gray(to_float01(im))
    pyr = [cur]
    while True:
        H, W = pyr[-1].shape[:2]
        if (min(H, W) // 2) < min_size:
            break
        if (max_levels is not None) and (len(pyr) >= max_levels):
            break
        pyr.append(downsample2(pyr[-1]))
    return pyr[::-1]  # return from coarsest -> finest

def _local_search(mov, ref, center=(0, 0), radius=5, method='ncc', crop_frac=0.1):
    """
    Search in a small window centered at 'center' on the current level.
    Returns (best_shift, best_score). Inputs are assumed grayscale float [0,1].
    """
    # prepare
    mov = to_float01(mov); ref = to_float01(ref)
    ref_crop = crop_border(ref, crop_frac)

    m = method.lower()
    if m == 'ncc':
        score_fn = ncc_score
        better = lambda s, best: (best is None) or (s > best)
    elif m == 'ssd':
        score_fn = ssd_score
        better = lambda s, best: (best is None) or (s < best)
    else:
        raise ValueError("method must be 'ncc' or 'ssd'")

    cy, cx = center
    best_score, best_shift = None, (cy, cx)
    for dy in range(cy - radius, cy + radius + 1):
        for dx in range(cx - radius, cx + radius + 1):
            a = crop_border(roll2(mov, dy, dx), crop_frac)
            s = score_fn(a, ref_crop)
            if better(s, best_score):
                best_score, best_shift = s, (dy, dx)
    return best_shift, best_score

def align_pyramid(moving, reference, base_radius=15, min_size=64, max_levels=None,
                  method='ncc', crop_frac=0.1, return_shift=False):
    """
    Coarse-to-fine alignment using a downsampling pyramid.
    - base_radius: search window radius at the finest level; upper levels shrink it.
    - min_size / max_levels: control pyramid depth.
    Returns aligned float image (and optionally (dy,dx), best_score at finest level).
    """
    # Build pyramids (coarse->fine), grayscale
    mov_pyr = build_pyramid(moving,  min_size=min_size, max_levels=max_levels)
    ref_pyr = build_pyramid(reference, min_size=min_size, max_levels=max_levels)
    if len(mov_pyr) != len(ref_pyr):
        L = min(len(mov_pyr), len(ref_pyr))
        mov_pyr, ref_pyr = mov_pyr[:L], ref_pyr[:L]

    # Coarse-to-fine search
    shift = (0, 0)
    best_score = None
    L = len(mov_pyr)
    for lvl in range(L):  # 0 = coarsest ... L-1 = finest
        movL, refL = mov_pyr[lvl], ref_pyr[lvl]

        # Project previous shift to this level (resolution doubled each step after the first)
        if lvl > 0:
            shift = (shift[0] * 2, shift[1] * 2)

        # Shrink search radius for coarser levels
        # at finest (lvl=L-1) => ~base_radius; at coarsest => ~base_radius / 2^(L-1)
        scale_down = 2 ** (L - 1 - lvl)
        radius = max(2, int(round(base_radius / scale_down)))

        shift, best_score = _local_search(
            movL, refL, center=shift, radius=radius,
            method=method, crop_frac=crop_frac
        )

    # Apply final shift on the original-resolution moving image
    aligned = roll2(to_float01(moving), shift[0], shift[1])

    if return_shift:
        return aligned, shift, best_score
    return aligned

