# 把兩張圖片混和
import numpy as np, cv2, os
from scipy.spatial import Delaunay

image_dir = "./HW3/images"
out_dir   = "./HW3/result"
os.makedirs(out_dir, exist_ok=True)

def imread_rgb(path):
    bgr = cv2.imread(path); assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

A = imread_rgb(os.path.join(image_dir,"imageA.png"))
B = imread_rgb(os.path.join(image_dir,"imageB.png"))
H,W = A.shape[:2]

ptsA = np.load(os.path.join(out_dir,"part1_points_A.npy")).astype(np.float32)
ptsB = np.load(os.path.join(out_dir,"part1_points_B.npy")).astype(np.float32)
# 一個每一列存三格頂點編號陣列的檔案
simp = np.load(os.path.join(out_dir,"part1_delaunay.npy"))
avg  = (ptsA + ptsB) / 2.0

# P1到P2跟P1到P3的向量外積，正值表示P1,P2,P3是逆時針
def signed_area(tri):
    (x1,y1),(x2,y2),(x3,y3) = tri
    return 0.5*((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
# 順時針的話交換P2跟P3
# Counter-Clockwise
def make_ccw(tri):
    if signed_area(tri) < 0:
        tri = tri.copy(); tri[[1,2]] = tri[[2,1]]
    return tri

# src是原本的圖，t_src是原本圖中三角形的三個頂點的座標大小(3, 2)
# dst是結果的圖，一開始是全黑的
# t_dst是結果三角形的三個頂點 用ptsA+ptsB/2得出的
def warp_triangle(src, dst, t_src, t_dst):
    t_src = make_ccw(t_src); t_dst = make_ccw(t_dst)
    # 接近共線的三角形面積會很接近0
    if abs(signed_area(t_dst)) < 1e-3: return
    # boundingRect 取得包住該三個頂點三角形的外接矩形 (x,y,w,h)
    # 就是找三角形三個座標中的min max的x y值再相減而已
    # x y是左上角座標，w h是寬高
    # 只處理小區塊，可以把三角形的座標改成相對於外接矩形左上角的座標，比較不會有大數字
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    # 撇除掉一些三角形超出邊界的情況
    if min(r1[2],r1[3],r2[2],r2[3]) <= 0: return
    # 把三角形的頂點座標改成相對於外接矩形左上角的座標
    t1r = np.float32([[t_src[i,0]-r1[0], t_src[i,1]-r1[1]] for i in range(3)])
    t2r = np.float32([[t_dst[i,0]-r2[0], t_dst[i,1]-r2[1]] for i in range(3)])
    # 產生一個mask，三角形區域是1，其餘是0
    mask = np.zeros((r2[3], r2[2]), np.float32)
    # LINE_AA讓三角形的邊界落在pixel中間的時候 會是一個中間值 不會是0或1
    cv2.fillConvexPoly(mask, np.int32(t2r), 1.0, lineType=cv2.LINE_AA)
    # r1[0]是x, r1[1]是y, r1[2]是w, r1[3]是h
    # 從輸入的圖片src_roi中取出該三角形外接矩形區域的部分
    src_roi = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_roi.size == 0: return
    # 目標:把來源三角形的頂點xi, yi 
    # 解方程式 a*xi+b*yi+c = xi' 跟 d*xi+e*yi+f = yi' 
    # 有三個x點剛好可解 a, b, c y點同理
    # M = [ [a b c], [d e f] ]
    M = cv2.getAffineTransform(t1r, t2r)
    # 用得到的矩陣M 變形原本的矩形 結果的長寬設定跟原本的矩形一樣
    # 如果超出邊界的話會填 0
    warped = cv2.warpAffine(src_roi, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 從要輸出的影像中取出該外接舉行的部分
    dst_roi = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    # dst_roi*(1-mask[...,None])保持輸出圖片在此三角形外的部分
    # warped*mask[...,None] 把三角形根據 mask 貼上去
    # mask[:, :, None] 變成 (h, w, 1) 這樣才能跟 (h, w, 3) 的圖片相乘
    dst_roi[:] = dst_roi*(1-mask[...,None]) + warped*mask[...,None]

def warp_image(img, src_pts, dst_pts, simp):
    out = np.zeros_like(img)
    # simp 每一列存三個頂點編號
    for s in simp:
        warp_triangle(img, out, src_pts[s], dst_pts[s])
    return out

warpA = warp_image(A, ptsA, avg, simp)
warpB = warp_image(B, ptsB, avg, simp)
# 找有標記的點的位置的凸包，會拿到是凸包上的點的index
hull_idx = cv2.convexHull(avg.astype(np.float32), returnPoints=False).flatten()
# 找凸包每個點的座標
hull = avg[hull_idx].astype(np.int32)
mask = np.zeros((H,W), np.float32)
# 把凸包的地方標記出來
cv2.fillConvexPoly(mask, hull, 1.0)
# (0, 0)自動調kernel # sigma=6
# 讓 mask邊界從1平滑過渡到0
mask = cv2.GaussianBlur(mask, (0,0), 6)  
# 把兩張圖分別轉換好的照片相加
mid = 0.5*warpA + 0.5*warpB
mid = mid * mask[...,None]
out = cv2.cvtColor((mid*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(out_dir,"part2_midway_face.jpg"), out)