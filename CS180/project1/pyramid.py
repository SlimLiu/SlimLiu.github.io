import numpy as np
import skimage.io as skio
from utils import split, to_float01, align_pyramid, crop_border

# 读入
plate = skio.imread("data/emir.tif")
B, G, R = split(plate)

#对齐
B_aln, sB, _ = align_pyramid(B, G, base_radius=20, min_size=64, method='ncc', return_shift=True)
R_aln, sR, _ = align_pyramid(R, G, base_radius=20, min_size=64, method='ncc', return_shift=True)
print("B→G:", sB, "  R→G:", sR)

#合成
B_aln, G, R_aln = to_float01(B_aln), to_float01(G), to_float01(R_aln)
RGB = np.dstack([R_aln, G, B_aln])

RGB = crop_border(RGB, 0.08)

skio.imsave("outputs/emir_colorized.jpg", (np.clip(RGB, 0, 1) * 255 + 0.5).astype(np.uint8))
