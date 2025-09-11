# 讓新的點是用以下的公式來調整，產生新的圖
# mean_pts 是 part4 算出來的所有人的臉的平均
# my_pts 拿 1a.jpg
# caric_pts=mean_pts+α⋅(my_pts−mean_pts)
import os, glob
import numpy as np
import cv2
from scipy.spatial import Delaunay
image_dir      = "./HW3/images"
all_images_dir = os.path.join(image_dir, "all_images")      
all_pts_dir    = os.path.join(image_dir, "all_images_pts")  
out_dir        = "./HW3/result"
os.makedirs(out_dir, exist_ok=True)
def imread_rgb(path):
    bgr = cv2.imread(path); 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

# 讀.pts檔
def read_pts(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    in_block = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("{"): in_block = True; continue
        if line.startswith("}"): break
        if in_block:
            p = line.split()
            if len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
    return np.stack([xs, ys], axis=1).astype(np.float32)   # (46,2)

# 加四個角點
def add_corners(pts, W, H):
    corners = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    return np.vstack([pts, corners])  # (46+4,2) = (50,2)

# 跟 part2 一樣的函式
def signed_area(tri):
    (x1,y1),(x2,y2),(x3,y3) = tri
    return 0.5*((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))

def make_ccw(tri):
    if signed_area(tri) < 0:
        tri = tri.copy(); tri[[1,2]] = tri[[2,1]]
    return tri

def warp_triangle(src, dst, t_src, t_dst):
    H, W = dst.shape[:2]
    t_src = make_ccw(t_src); t_dst = make_ccw(t_dst)
    if abs(signed_area(t_dst)) < 1e-3:   
        return
    r1 = cv2.boundingRect(np.float32([t_src]))  # (x,y,w,h)
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if min(r1[2], r1[3], r2[2], r2[3]) <= 0:
        return
    t1r = np.float32([[t_src[i,0]-r1[0], t_src[i,1]-r1[1]] for i in range(3)])
    t2r = np.float32([[t_dst[i,0]-r2[0], t_dst[i,1]-r2[1]] for i in range(3)])
    src_roi = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_roi.size == 0:
        return
    mask = np.zeros((r2[3], r2[2]), np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2r), 1.0, lineType=cv2.LINE_AA)
    M = cv2.getAffineTransform(t1r, t2r)
    warped = cv2.warpAffine(
        src_roi, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101  # 避免黑邊
    )

    # 可能超出影像，裁切掉
    x0, y0, w2, h2 = r2
    x1, y1 = x0 + w2, y0 + h2
    cx0 = max(0, x0); cy0 = max(0, y0)
    cx1 = min(W, x1); cy1 = min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    # 裁切後的偏移
    dx, dy = cx0 - x0, cy0 - y0
    ww, hh = cx1 - cx0, cy1 - cy0

    dst_roi = dst[cy0:cy1, cx0:cx1]
    warped_crop = warped[dy:dy+hh, dx:dx+ww]
    mask_crop   = mask[dy:dy+hh, dx:dx+ww]

    if dst_roi.size == 0 or warped_crop.size == 0 or mask_crop.size == 0:
        return

    dst_roi[:] = dst_roi*(1 - mask_crop[...,None]) + warped_crop*mask_crop[...,None]

def warp_image(img, src_pts, dst_pts, simp):
    out = np.zeros_like(img)
    for s in simp:
        warp_triangle(img, out, src_pts[s], dst_pts[s])
    return out

def main():
    pairs = []
    for jp in sorted(glob.glob(os.path.join(all_images_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(jp))[0]
        ptp  = os.path.join(all_pts_dir, f"{name}.pts")
        if os.path.exists(ptp):
            pairs.append((jp, ptp))
    A0 = imread_rgb(pairs[0][0])
    H, W = A0.shape[:2]

    # 讀全部照片取平均
    pts_list = []
    for _, p in pairs:
        pts = read_pts(p)
        pts = add_corners(pts, W, H)
        pts_list.append(pts)
    pts_list = np.stack(pts_list, axis=0) # (N,50,2)
    mean_pts = pts_list.mean(axis=0) # (50,2)
    simp = Delaunay(mean_pts).simplices  # (T,3)

    # 1a.jpg當例子
    my_img = imread_rgb(os.path.join(all_images_dir, "1a.jpg"))
    my_pts = add_corners(read_pts(os.path.join(all_pts_dir, "1a.pts")), W, H)

    # 用不同的alpha值
    alphas = [-1, 0.5, 1, 1.5, 2.0]  
    for a in alphas:
        caric_pts = mean_pts + a*(my_pts - mean_pts)  
        caric_img = warp_image(my_img, my_pts, caric_pts, simp)
        out_path = os.path.join(out_dir, f"part5_caricature_alpha{a}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor((caric_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
