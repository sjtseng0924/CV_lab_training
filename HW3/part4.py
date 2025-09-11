# 資料來源:https://fei.edu.br/~cet/facedatabase.html
# 用1-100張照片以及他們對應的標記點
import os, glob
import numpy as np
import cv2
from scipy.spatial import Delaunay

image_dir = "./HW3/images"
all_images = os.path.join(image_dir, "all_images")       
all_images_pts = os.path.join(image_dir, "all_images_pts")   
out_dir = "./HW3/result"
os.makedirs(out_dir, exist_ok=True)

def imread_rgb(path):
    bgr = cv2.imread(path); 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
# 讀.pts檔 .pts檔案可讀 檔案較大
def read_pts(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    in_block = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("{"):
            in_block = True; continue
        if line.startswith("}"):
            break
        if in_block:
            p = line.split()
            if len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
    pts = np.stack([xs, ys], axis=1).astype(np.float32)   # (46,2)
    return pts

# part2 中的一些函式 用來把原本的人臉變成指定的形狀
def signed_area(tri):
    (x1,y1),(x2,y2),(x3,y3) = tri
    return 0.5*((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))

def make_ccw(tri):
    if signed_area(tri) < 0:
        tri = tri.copy(); tri[[1,2]] = tri[[2,1]]
    return tri

def warp_triangle(src, dst, t_src, t_dst):
    t_src = make_ccw(t_src); t_dst = make_ccw(t_dst)
    if abs(signed_area(t_dst)) < 1e-3: return

    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if min(r1[2],r1[3],r2[2],r2[3]) <= 0: return

    t1r = np.float32([[t_src[i,0]-r1[0], t_src[i,1]-r1[1]] for i in range(3)])
    t2r = np.float32([[t_dst[i,0]-r2[0], t_dst[i,1]-r2[1]] for i in range(3)])

    mask = np.zeros((r2[3], r2[2]), np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2r), 1.0, lineType=cv2.LINE_AA)

    src_roi = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_roi.size == 0: return

    M = cv2.getAffineTransform(t1r, t2r)
    warped = cv2.warpAffine(src_roi, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dst_roi = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst_roi[:] = dst_roi*(1-mask[...,None]) + warped*mask[...,None]

def warp_image(img, src_pts, dst_pts, simp):
    out = np.zeros_like(img)
    for s in simp:
        warp_triangle(img, out, src_pts[s], dst_pts[s])
    return out

# 加四個角的點
def add_corners(pts, W, H):
    corners = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    return np.vstack([pts, corners])  # (46+4, 2)

# 把jpg跟pst檔案配對
def collect_pairs(img_dir, pts_dir):
    pairs = []
    for jp in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(jp))[0]   
        ptp  = os.path.join(pts_dir, f"{name}.pts")
        if os.path.exists(ptp):
            pairs.append((jp, ptp))
    return pairs

def main():
    pairs = collect_pairs(all_images, all_images_pts)
    # 讀第一張決定大小
    A0 = imread_rgb(pairs[0][0])
    H, W = A0.shape[:2]
    imgs = []
    pts_list = []
    for img_path, pts_path in pairs:
        img = imread_rgb(img_path)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        pts = read_pts(pts_path) # (46,2)
        pts = add_corners(pts, W, H) # (50,2) 加入角落點
        imgs.append(img)
        pts_list.append(pts)
    # imgs是list 疊成numpy array
    imgs = np.stack(imgs, axis=0) # (N,H,W,3)
    pts_list = np.stack(pts_list, axis=0) # (N,50,2)
    N = pts_list.shape[0]
    # 沿著第0維平均，讓他變成(50, 2)
    avg = pts_list.mean(axis=0)
    # 三角形頂點的編號        
    simp = Delaunay(avg).simplices     

    # 初始值 每warp一張圖完就加到這個變數中
    acc = np.zeros_like(A0)
    for i in range(N):
        # 每一張照片的比重占1/N
        warped = warp_image(imgs[i], pts_list[i], avg, simp)
        acc += warped / N
    mean_face = np.clip(acc, 0, 1)
    # 存下100張人臉的平均臉的結果
    cv2.imwrite(
        os.path.join(out_dir, "part4_mean_face.jpg"),
        cv2.cvtColor((mean_face*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    # 拿第一張照片來當作例子
    my_img_path = os.path.join(all_images, "1a.jpg")
    my_pts_path = os.path.join(all_images_pts, "1a.pts")
    if os.path.exists(my_img_path) and os.path.exists(my_pts_path):
        my_img = imread_rgb(my_img_path)
        my_pts = read_pts(my_pts_path)
        my_pts = add_corners(my_pts, W, H)
        # 把第一張照片的臉型轉成平均臉型的形狀
        mine_to_mean = warp_image(my_img, my_pts, avg, simp)
        cv2.imwrite(
            os.path.join(out_dir, "part4_my_face_to_mean_shape.jpg"),
            cv2.cvtColor((mine_to_mean*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        # 把轉成平均臉型的形狀轉成第一張照片的臉型
        mean_to_mine = warp_image(mean_face, avg, my_pts, simp)
        cv2.imwrite(
            os.path.join(out_dir, "part4_mean_face_to_my_shape.jpg"),
            cv2.cvtColor((mean_to_mine*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

if __name__ == "__main__":
    main()
