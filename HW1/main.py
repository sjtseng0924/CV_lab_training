import os
import time
import csv
import numpy as np
from skimage import io, transform
from skimage.filters import sobel
# 將像素轉成 [0,1] 的 float
def im2float01(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    img = img.astype(np.float32)
    # img.max() 找出圖片最大值
    vmax = img.max() if np.isfinite(img.max()) else 1.0
    # 避免除以 0
    return img / (vmax if vmax > 0 else 1.0)

# 照片是垂直堆疊的 BGR
def split_channels_bgr_stacked(img):
    # shape[0] 照片是高度
    H = img.shape[0] // 3
    B = img[0:H, :]
    G = img[H:2*H, :]
    R = img[2*H:3*H, :]
    return R, G, B

# 上下左右各裁掉5%的邊界
def crop_ratio(img, ratio=0.05):
    # 拿 shape[0跟1] 是他的高跟寬
    h, w = img.shape[:2]
    dh = int(h * ratio)
    dw = int(w * ratio)
    if dh == 0 or dw == 0:
        return img
    return img[dh:h-dh, dw:w-dw]

# Normalized Cross-Correlation 回傳這兩塊圖在這個位置下的相似度分數，越高越好
# 用NCC的原因是因為它會標準化，可能有些東西的亮度比較亮，不然如果在同一個範圍可以直接用相減的結果去比較
def ncc(a, b, eps=1e-8):
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.sum(a * b) / denom)

# 把三個顏色的平均拉到同一個亮度，因為第三維axis=2是亮度，所以axis=(0, 1)
def white_balance_gray_world(rgb):
    # axis=(0, 1)是np.mean(rgb[:, :, 0]) np.mean(rgb[:, :, 1]) ...
    avg = np.mean(rgb, axis=(0, 1))
    scale = avg.mean() / (avg + 1e-8)
    # np.clip 把值限制在 0 到 1 之間
    wb = np.clip(rgb * scale, 0, 1)
    return wb

# compare圖左右並排
def make_side_by_side(img_left, img_right):
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    H = max(h1, h2)

    def pad_to_h(img, H):
        h, w = img.shape[:2]
        if h == H:
            return img
        pad_top = (H - h) // 2
        pad_bottom = H - h - pad_top
        if img.ndim == 3:
            pad_cfg = ((pad_top, pad_bottom), (0, 0), (0, 0))
        else:
            pad_cfg = ((pad_top, pad_bottom), (0, 0))
        return np.pad(img, pad_cfg, mode='edge')

    L = pad_to_h(img_left, H)
    R = pad_to_h(img_right, H)
    return np.concatenate([L, R], axis=1)

def preprocess_for_metric(img, use_edges=True):
    if use_edges:
        # sobel 會回傳邊緣強度圖(越靠近1越可能是邊緣)，值通常在0-1之間
        # 是靠img的亮度差去分析的
        return sobel(img)
    return img

def align_single_scale(mov, ref, search=15, crop=0.05, use_edges=True):
    """
    單層窮舉對齊：
    在 [-search, search] 內找 NCC 最大時候的位移量
    用 RG 去對齊 B
    mov：移動的影像 RG 
    ref：參考影像用 B
    """
    mov_p = preprocess_for_metric(mov, use_edges=use_edges)
    ref_p = preprocess_for_metric(ref, use_edges=use_edges)

    # 目前最好的ncc分數跟位移
    best = (-999.0, 0, 0)  # (score, dx, dy)
    for dx in range(-search, search + 1):
        for dy in range(-search, search + 1):
            # np.roll 是循環位移
            shifted = np.roll(np.roll(mov_p, dx, axis=0), dy, axis=1)
            s = ncc(crop_ratio(shifted, crop), crop_ratio(ref_p, crop))
            if s > best[0]:
                best = (s, dx, dy)
    # 回傳最高分數時候的位移量
    return (best[1], best[2])

def pyramid_align(mov, ref,
                  search=15, refine_radius=2,
                  min_size=120, use_edges=True, crop=0.05):
    """
    金字塔對齊（遞迴）：
    尺寸小於 min_size 時改用單層窮舉
    否則將影像縮小 1/2 對齊，再放大兩倍後在附近做小半徑refine
    """
    h, w = mov.shape[:2]
    if min(h, w) < min_size:
        return align_single_scale(mov, ref, search=search, crop=crop, use_edges=use_edges)

    # 先對齊較小尺寸
    mov_small = transform.rescale(mov, 0.5, anti_aliasing=False, channel_axis=None)
    ref_small = transform.rescale(ref, 0.5, anti_aliasing=False, channel_axis=None)
    dx, dy = pyramid_align(mov_small, ref_small,
                           search=search, refine_radius=refine_radius,
                           min_size=min_size, use_edges=use_edges, crop=crop)
    dx *= 2
    dy *= 2

    # 在 (dx,dy) 附近小範圍微調
    mov_p = preprocess_for_metric(mov, use_edges=use_edges)
    ref_p = preprocess_for_metric(ref, use_edges=use_edges)
    best = (-999.0, dx, dy)
    for ddx in range(dx - refine_radius, dx + refine_radius + 1):
        for ddy in range(dy - refine_radius, dy + refine_radius + 1):
            shifted = np.roll(np.roll(mov_p, ddx, axis=0), ddy, axis=1)
            s = ncc(crop_ratio(shifted, crop), crop_ratio(ref_p, crop))
            if s > best[0]:
                best = (s, ddx, ddy)
    return (best[1], best[2])

def apply_shift(img, shift):
    dx, dy = shift
    return np.roll(np.roll(img, dx, axis=0), dy, axis=1)

def process_one(image_path,
                out_dir=os.path.join(os.path.dirname(__file__), "results"),
                use_pyramid=True,
                search=15,
                refine_radius=2,
                min_size=120,
                use_edges=False,
                crop_for_score=0.05):
    t0 = time.time()
    img_raw = io.imread(image_path)
    img = im2float01(img_raw)
    R, G, B = split_channels_bgr_stacked(img)
    base = B  
    # baseline
    before_rgb = np.clip(np.dstack([R, G, B]), 0, 1)
    # G 和 R 對齊到 B
    align_func = pyramid_align if use_pyramid else align_single_scale
    shift_G = align_func(G, base, search=search, refine_radius=refine_radius,
                         min_size=min_size, use_edges=use_edges, crop=crop_for_score)
    shift_R = align_func(R, base, search=search, refine_radius=refine_radius,
                         min_size=min_size, use_edges=use_edges, crop=crop_for_score)
    G_aligned = apply_shift(G, shift_G)
    R_aligned = apply_shift(R, shift_R)

    after_rgb = np.dstack([R_aligned, G_aligned, B])
    after_rgb = white_balance_gray_world(after_rgb)

    # 對齊後照片
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(image_path))[0]
    out_color = os.path.join(out_dir, f"{fname}_color.jpg")
    io.imsave(out_color, (after_rgb * 255).astype(np.uint8))

    # 未對齊 vs 對齊照片
    comp = make_side_by_side((before_rgb * 255).astype(np.uint8),
                             (after_rgb * 255).astype(np.uint8))
    out_comp = os.path.join(out_dir, f"{fname}_compare.jpg")
    io.imsave(out_comp, comp)
    
    elapsed = time.time() - t0
    print(f"[OK] {fname} | ShiftG={shift_G}, ShiftR={shift_R} | {elapsed:.2f}s")
    return fname, shift_G, shift_R, elapsed

def main():
    script_dir = os.path.dirname(__file__)  # 取得當前目錄
    image_dir = os.path.join(script_dir, "images")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))]
    files.sort()

    print(f"共 {len(files)} 張圖，開始處理…")
    rows = [("filename", "shiftG_dx", "shiftG_dy", "shiftR_dx", "shiftR_dy", "seconds")]

    for f in files:
        path = os.path.join(image_dir, f)
        name, shiftG, shiftR, sec = process_one(path)
        rows.append((name, shiftG[0], shiftG[1], shiftR[0], shiftR[1], f"{sec:.2f}"))

    csv_path = os.path.join(script_dir, "results", "shifts.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)  
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
    print(f"位移已寫入：{csv_path}")


if __name__ == "__main__":
    main()
