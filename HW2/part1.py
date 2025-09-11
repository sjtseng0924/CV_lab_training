import cv2
import numpy as np
from scipy.signal import convolve2d
import os

def imread_gray(path):
    # cv2.IMREAD_GRAYSCALE 把圖片轉成灰階
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def save_image_gray(path, img):
    cv2.imwrite(path, (img * 255).clip(0, 255).astype(np.uint8))

def finite_difference_edge(img, threshold_ratio=0.25):
    # Dx是一個[1, -1]的陣列，是一個filter，會拿去跑整張圖，得到右邊像素-左邊像素的值(會反轉在做)
    # 值突然差很大的地方，有可能是邊緣
    Dx = np.array([[1, -1]], dtype=np.float32)
    # 轉置看垂直效果
    Dy = Dx.T
    # 對整張圖片套用Dx運算，mode='same'表示輸出大小和原圖一樣
    # boundary='symm'表示邊界處的像素是對稱延伸，假裝旁邊那格跟自己一樣
    # 可以用一維水平跟垂直的陣列去做兩次，會比一個二維的陣列去convolve快
    gx = convolve2d(img, Dx, mode='same', boundary='symm')
    gy = convolve2d(img, Dy, mode='same', boundary='symm')
    # 平方開根號，數值越大代表變化越大，越可能是邊緣
    grad = np.sqrt(gx**2 + gy**2)
    edge = (grad > grad.max() * threshold_ratio).astype(np.uint8)
    return edge

def dog_edge(img, ksize=7, sigma=1.0, threshold_ratio=0.25):
    Dx = np.array([[1, -1]], dtype=np.float32)
    Dy = Dx.T
    # 用高斯函數生成 filter，sigma 越大會越模糊，ksize 是該 filter 矩陣的大小
    # ksize 必須是奇數，因為要有中心
    # 高斯函數是無限長的，但filter只取ksize大小，因此sigma大時，ksize也要夠大
    # ksize 大概會取 sigma * 6 + 1，大約可以包含 99% 的權重和
    # 產生的1D矩陣總和會是1
    # sigma 很大但 ksize 很小會導致模糊不夠
    G1D = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    # G用來模糊圖片的filter
    G = G1D @ G1D.T
    # mode='full'輸出所有跑過的區域，輸出會比原輸入大
    # 創造先模糊後差分的 filter
    DoGx = convolve2d(G, Dx, mode='full')
    DoGy = convolve2d(G, Dy, mode='full')
    gx = convolve2d(img, DoGx, mode='same', boundary='symm')
    gy = convolve2d(img, DoGy, mode='same', boundary='symm')
    grad = np.sqrt(gx**2 + gy**2)
    edge = (grad > grad.max() * threshold_ratio).astype(np.uint8)
    return edge

def main():
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "images", "cameraman.png")
    result_dir = os.path.join(script_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    img = imread_gray(image_path)
    edge_fd = finite_difference_edge(img)
    save_image_gray(os.path.join(result_dir, "part1_fd_edge.png"), edge_fd)
    edge_dog = dog_edge(img)
    save_image_gray(os.path.join(result_dir, "part1_dog_edge.png"), edge_dog)

if __name__ == "__main__":
    main()
