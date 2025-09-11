import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, skimage.io as skio

image_dir = "./HW3/images"
out_dir   = "./HW3/result"
os.makedirs(out_dir, exist_ok=True)

imgA = skio.imread(os.path.join(image_dir,"imageA.png"))
imgB = skio.imread(os.path.join(image_dir,"imageB.png"))
H, W = imgA.shape[:2]
assert imgA.shape[:2] == imgB.shape[:2]

N_LM = 43
def get_points(img, n, title):
    plt.figure(); plt.imshow(img); plt.title(title)
    pts = plt.ginput(n, timeout=0); plt.close()
    return np.array(pts, dtype=np.float32)

ptsA = get_points(imgA, N_LM, f"Image A")
ptsB = get_points(imgB, N_LM, f"Image B")

def boundary_points(w,h):
    return np.array([
        (0,0),(w-1,0),(w-1,h-1),(0,h-1),
        (w//2,0),(w-1,h//2),(w//2,h-1),(0,h//2)
    ], dtype=np.float32)

bnd = boundary_points(W,H)
ptsA = np.vstack([ptsA, bnd])
ptsB = np.vstack([ptsB, bnd])

# 用中間的點來做delaunay
# delaunay是讓點連成三角形的演算法
# 三角剖分中的任意一個三角形，它的外接圓不會包含任何其他的輸入點
# simplices代表最簡單的幾何單元 三角形
avg  = (ptsA + ptsB) / 2.0
tri  = Delaunay(avg).simplices

np.save(os.path.join(out_dir,"part1_points_A.npy"), ptsA.astype(np.float32))
np.save(os.path.join(out_dir,"part1_points_B.npy"), ptsB.astype(np.float32))
np.save(os.path.join(out_dir,"part1_delaunay.npy"), tri)
# 把產生出來的三角形的結果畫在imageA上
plt.figure(); plt.imshow(imgA)
plt.triplot(ptsA[:,0], ptsA[:,1], tri, color='cyan')
plt.scatter(ptsA[:,0], ptsA[:,1], c='r', s=10)
plt.title("Triangulation topology from average, drawn on A")
plt.savefig(os.path.join(out_dir,"part1_triangulation_A.jpg"), dpi=150)
