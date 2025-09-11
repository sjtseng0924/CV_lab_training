# 把照片的融合 變成一個漸進式的gif
import os
import numpy as np
import cv2
from scipy.spatial import Delaunay
import imageio.v2 as imageio  

# 跟3-2的合成一樣的函式
def signed_area(tri): 
    (x1,y1),(x2,y2),(x3,y3) = tri
    return 0.5*((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))

def make_ccw(tri):
    tri = tri.copy()
    if signed_area(tri) < 0:
        tri[[1,2]] = tri[[2,1]]
    return tri

def warp_triangle(src, dst, t_src, t_dst):
    t_src = make_ccw(t_src); t_dst = make_ccw(t_dst)
    if abs(signed_area(t_dst)) < 1e-3:
        return
    r1 = cv2.boundingRect(np.float32([t_src]))  # (x,y,w,h)
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if min(r1[2],r1[3],r2[2],r2[3]) <= 0:
        return
    t1r = np.float32([[t_src[i,0]-r1[0], t_src[i,1]-r1[1]] for i in range(3)])
    t2r = np.float32([[t_dst[i,0]-r2[0], t_dst[i,1]-r2[1]] for i in range(3)])
    mask = np.zeros((r2[3], r2[2]), np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2r), 1.0, lineType=cv2.LINE_AA)
    src_roi = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_roi.size == 0:
        return
    M = cv2.getAffineTransform(t1r, t2r)
    warped = cv2.warpAffine(
        src_roi, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    dst_roi = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst_roi[:] = dst_roi*(1 - mask[...,None]) + warped*mask[...,None]

def warp_image(img, src_pts, dst_pts, simplices):
    out = np.zeros_like(img)
    for s in simplices:
        warp_triangle(img, out, src_pts[s], dst_pts[s])
    return out

def morph(im1, im2, im1_pts, im2_pts, tri_simplices, warp_frac, dissolve_frac):
    # 不是用0.5跟0.5來算的中間的point
    inter_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts
    # 算出中間的那些點後，用3-2的方法去產生兩個圖分別對齊這個形狀的結果
    warp1 = warp_image(im1, im1_pts, inter_pts, tri_simplices)
    warp2 = warp_image(im2, im2_pts, inter_pts, tri_simplices)
    # dissolve_frac看哪一張圖要加的比較多(控制顏色)
    morphed = (1 - dissolve_frac) * warp1 + dissolve_frac * warp2
    return np.clip(morphed, 0, 1)

if __name__ == "__main__":
    image_dir = "./HW3/images"
    result_dir = "./HW3/result"
    os.makedirs(result_dir, exist_ok=True)
    def imread_rgb_float(path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32)/255.0
    im1 = imread_rgb_float(os.path.join(image_dir, "imageA.png"))
    im2 = imread_rgb_float(os.path.join(image_dir, "imageB.png"))
    H, W = im1.shape[:2]
    im1_pts = np.load(os.path.join(result_dir, "part1_points_A.npy")).astype(np.float32)
    im2_pts = np.load(os.path.join(result_dir, "part1_points_B.npy")).astype(np.float32)
    tri_path = os.path.join(result_dir, "part1_delaunay.npy")
    tri_simplices = np.load(tri_path)

    # t從0到1
    num_frames = 31  # 做31張
    frames = []
    for i, t in enumerate(np.linspace(0, 1, num_frames)):
        frame = morph(
            im1, im2, im1_pts, im2_pts, tri_simplices,
            warp_frac=t,           # 混和兩張圖片的點的比例值
            dissolve_frac=t,       # 顏色比例值
        )
        # imageio要uint8
        frames.append((frame*255).astype(np.uint8))
        print(f"frame {i+1}/{num_frames} done (t={t:.3f})")
    gif_path = os.path.join(result_dir, "part3_morph.gif")
    # 把frames存成gif
    imageio.mimsave(gif_path, frames, duration=0.05)  
