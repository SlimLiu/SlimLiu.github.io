import numpy as np
from pathlib import Path
import skimage.io as skio

from utils import align, split, to_gray, to_float01,crop_border

# 读入
imname = 'data/cathedral.jpg'
im = skio.imread(imname)

if im.ndim == 3 and im.shape[2] == 1:
    im = im[..., 0]

# 切分
b, g, r = split(im)

b_aligned, shift_b, score_b = align(b, g, search_radius=15, method='ncc', return_shift=True)
r_aligned, shift_r, score_r = align(r, g, search_radius=15, method='ncc', return_shift=True)

print("B shift:", shift_b, "score:", score_b)
print("R shift:", shift_r, "score:", score_r)

# 合成

def match_mean(x, ref, frac=0.1):
    xc = crop_border(x,   frac)
    rc = crop_border(ref, frac)
    s = (rc.mean() + 1e-8) / (xc.mean() + 1e-8)
    y = np.clip(x * s, 0.0, 1.0)
    return y

g = to_float01(g)
b_aligned = to_float01(b_aligned)
r_aligned = to_float01(r_aligned)
im_out = np.dstack([r_aligned, g, b_aligned])
im_out = crop_border(im_out, 0.08)

out_path = Path('outputs/cathedral_colorized.jpg')
out_path.parent.mkdir(parents=True, exist_ok=True)

# 保险起见裁到 [0,1]
im_out = np.clip(im_out, 0.0, 1.0)
skio.imsave(out_path.as_posix(), (im_out * 255.0 + 0.5).astype(np.uint8))
