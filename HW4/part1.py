import matplotlib
matplotlib.use("TkAgg")  
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

BASE_DIR = Path(__file__).resolve().parent
IM_DIR   = BASE_DIR / "images"      
OUT_DIR  = BASE_DIR / "results"     
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_PATH   = IM_DIR / "lobby0.jpg"
CENTER_PATH = IM_DIR / "lobby1.jpg"
RIGHT_PATH  = IM_DIR / "lobby2.jpg"
NUM_POINTS  = 6 

def load_img(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def save_img(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

# 把點正規化，把標準差調乘根號二，然後回傳
def _normalize_points(pts: np.ndarray):
    # Hartley的論文提出
    # 算出所有點的中心點，讓中心點變成原點(0, 0)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0).mean() + 1e-12
    # 因為DLT的誤差很大，想讓每個點到新原點的平均距離是根號2，等於每一個維度大概是1
    s = (2.0**0.5) / std
    # T*[x, y, 1]^T = [s*x - s*mean_x, s*y - s*mean_y, 1]^T
    # s*(x - mean_x) => 等於把x標準化之後再乘以根號2 -> 讓新的標準叉是根號二，中心點是原本的中心點(但平移去原點)
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]], dtype=np.float64)
    # 把點轉成(x, y, 1)，方便直接乘矩陣
    # np.ones((pts.shape[0], 1)) 產生一個全是1的向量，點的數量跟pts一樣多
    # hstack是水平拼接，原本是(N, 2)變成(N, 3)，(x, y)變成(x, y, 1)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    # @是矩陣乘法
    # .T是轉置矩陣
    ptsn = (T @ pts_h.T).T
    # 回傳所有列跟前兩欄，(N, 2)
    return ptsn[:, 0:2], T

# 要算Direct Linear Transform(DLT)
# [u, v, 1]^T 正比 H * [x, y, 1]^T，要解其中的H矩陣
# 正比的原因是這個點乘任意非零常數都還是同一個點
# H有9個未知數，但因為比例不重要，所以自由度只有8
# 每一對點可以提供兩個方程式(u=h11x+h12y..., v=h21x+...)，所以至少要4對點才能解出H
def computeH(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # 兩對對應的點的個數
    N = p1.shape[0]
    A = []
    for i in range(N):
        x, y = p1[i]; u, v = p2[i]
        # 那兩個方程式展開之後的係數，把x, y用u, v跟h11, h12...h33表示
        # 移向等號左邊變成0
        # u*h31*x + u*h32*y + u*h33 - h11*x - h12*y - h13 = 0
        # v*h31*x + v*h32*y + v*h33 - h21*x - h22*y - h23 = 0
        # 以下的九個數字分別是h11, h12,...,h33的係數
        A.append([0,0,0, -x,-y,-1, v*x, v*y, v])
        A.append([x,y,1,  0, 0, 0, -u*x,-u*y,-u])
    # 把 H 攤平變成一個[9*1]的矩陣h
    # Ah = 0
    A = np.asarray(A, dtype=np.float64)
    # SVD分解把矩陣拆成單位正交矩陣U、V和對角矩陣S
    # A = U*S*V^T
    _,_,VT = np.linalg.svd(A)
    H = VT[-1,:].reshape(3,3)
    if abs(H[2,2]) > 1e-12: H /= H[2,2]
    return H

def computeH_norm(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    p1n, T1 = _normalize_points(p1.astype(np.float64))
    p2n, T2 = _normalize_points(p2.astype(np.float64))
    Hn = computeH(p1n, p2n)
    H  = np.linalg.inv(T2) @ Hn @ T1
    if abs(H[2,2]) > 1e-12: H /= H[2,2]
    return H

# ================== Warping（Inverse + 雙線性取樣） ==================
def apply_homography(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    dst = (H @ pts_h.T).T
    return dst[:, :2] / (dst[:, 2:] + 1e-12)

def image_corners_wh(w: int, h: int):
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float64)

def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        out = [ndimage.map_coordinates(img[:,:,c], [y,x], order=1, mode='constant', cval=0.0)
               for c in range(img.shape[2])]
        return np.stack(out, axis=-1)
    return ndimage.map_coordinates(img, [y,x], order=1, mode='constant', cval=0.0)

def warp_image_inverse(src: np.ndarray, H: np.ndarray, out_bounds=None):
    h, w = src.shape[:2]
    dst_corners = apply_homography(image_corners_wh(w,h), H)
    if out_bounds is None:
        xmin, ymin = np.floor(dst_corners.min(axis=0)).astype(int)
        xmax, ymax = np.ceil(dst_corners.max(axis=0)).astype(int)
    else:
        (xmin,xmax),(ymin,ymax) = out_bounds

    X, Y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    src_xy = apply_homography(XY, np.linalg.inv(H))
    xs = src_xy[:,0].reshape(Y.shape)
    ys = src_xy[:,1].reshape(Y.shape)

    warped = bilinear_sample(src, xs, ys)
    alpha  = ((xs>=0)&(xs<=w-1)&(ys>=0)&(ys<=h-1)).astype(np.float32)
    alpha  = np.repeat(alpha[...,None], 3, axis=2)
    return warped, alpha, (xmin, xmax, ymin, ymax)

# ================== Blending & Stitch ==================
def feather_alpha(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3: mask = mask[...,0]
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=10.0)
    blurred = blurred / (blurred.max() + 1e-12)
    return np.repeat(blurred[...,None], 3, axis=2).astype(np.float32)

def blend_pair(base_img, base_bounds, warped, alpha, warped_bounds):
    bxmin,bxmax,bymin,bymax = base_bounds
    wxmin,wxmax,wymin,wymax = warped_bounds
    xmin, xmax = min(bxmin, wxmin), max(bxmax, wxmax)
    ymin, ymax = min(bymin, wymin), max(bymax, wymax)

    H = ymax - ymin + 1
    W = xmax - xmin + 1
    out = np.zeros((H,W,3), dtype=np.float32)
    out_alpha = np.zeros((H,W,3), dtype=np.float32)

    # base
    by0, bx0 = bymin - ymin, bxmin - xmin
    out[by0:by0+(bymax-bymin+1), bx0:bx0+(bxmax-bxmin+1)] += base_img
    out_alpha[by0:by0+(bymax-bymin+1), bx0:bx0+(bxmax-bxmin+1)] += 1.0

    # warped with feather
    wy0, wx0 = wymin - ymin, wxmin - xmin
    walpha = feather_alpha((alpha[...,0] > 0).astype(np.float32))
    sl = np.s_[wy0:wy0+(wymax-wymin+1), wx0:wx0+(wxmax-wxmin+1)]
    out[sl] = out[sl]*(1-walpha) + warped*walpha
    out_alpha[sl] = np.maximum(out_alpha[sl], walpha)

    return np.clip(out,0,1), (xmin,xmax,ymin,ymax)

def stitch_two(ref_img, src_img, H_src_to_ref):
    rh, rw = ref_img.shape[:2]
    ref_bounds = (0, rw-1, 0, rh-1)
    warped, alpha, w_bounds = warp_image_inverse(src_img, H_src_to_ref)
    return blend_pair(ref_img, ref_bounds, warped, alpha, w_bounds)

def stitch_three(center, left, right, H_left_to_center, H_right_to_center):
    ch, cw = center.shape[:2]
    mosaic, bounds = center.copy(), (0, cw-1, 0, ch-1)
    warpedL, alphaL, bL = warp_image_inverse(left, H_left_to_center)
    mosaic, bounds = blend_pair(mosaic, bounds, warpedL, alphaL, bL)
    warpedR, alphaR, bR = warp_image_inverse(right, H_right_to_center)
    mosaic, bounds = blend_pair(mosaic, bounds, warpedR, alphaR, bR)
    return mosaic, bounds

# 標記左中右圖片的點，左邊跟中間圖對應6個點，右邊跟中間圖對應6個點
def collect_points_pair(imgA, imgB, n=6):
    print(f"lobby0")
    figA = plt.figure(figsize=(8,8)); plt.imshow(imgA); plt.axis('off')
    plt.title(f"lobby1"); plt.show(block=False)
    ptsA = plt.ginput(n=n, timeout=0); plt.close(figA)

    print(f"lobby2")
    figB = plt.figure(figsize=(8,8)); plt.imshow(imgB); plt.axis('off')
    plt.title(f"lobby1"); plt.show(block=False)
    ptsB = plt.ginput(n=n, timeout=0); plt.close(figB)

    return np.array(ptsA, dtype=np.float64), np.array(ptsB, dtype=np.float64)

def main():
    # 讀圖
    imgL = load_img(LEFT_PATH)
    imgC = load_img(CENTER_PATH)
    imgR = load_img(RIGHT_PATH)

    ptsL, ptsC = collect_points_pair(imgL, imgC, n=NUM_POINTS)
    H_LC = computeH_norm(ptsL, ptsC)
    mosaic_LC, _ = stitch_two(imgC, imgL, H_LC)
    save_img(OUT_DIR / "mosaic_left_center.png", mosaic_LC)
    
    ptsR, ptsC2 = collect_points_pair(imgR, imgC, n=NUM_POINTS)
    H_RC = computeH_norm(ptsR, ptsC2)
    mosaic_RC, _ = stitch_two(imgC, imgR, H_RC)
    save_img(OUT_DIR / "mosaic_right_center.png", mosaic_RC)
    
    # 三張整合
    mosaic_3, _ = stitch_three(imgC, imgL, imgR, H_LC, H_RC)
    save_img(OUT_DIR / "mosaic_lcr.png", mosaic_3)

if __name__ == "__main__":
    main()