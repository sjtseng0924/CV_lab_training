# 把一張照片的高頻跟另一張照片的低頻合起來變成一張新的照片
import cv2
import numpy as np
import os

def read_image_gray32(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def save_image_gray(path, img):
    cv2.imwrite(path, (img * 255).clip(0, 255).astype(np.uint8))

# 模糊照片
def gaussian_blur(img, ksize=15, sigma=3.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

# 用原照片剪掉糊化照片，得到高頻部分
def get_high_frequency(img, ksize=15, sigma=3.0):
    low_pass = gaussian_blur(img, ksize, sigma)
    high_pass = img - low_pass
    return high_pass

def make_hybrid_image(low_img, high_img, ksize=15, sigma=3.0, alpha=0.5):
    low_freq = gaussian_blur(low_img, ksize, sigma)
    high_freq = get_high_frequency(high_img, ksize, sigma)
    hybrid = (low_freq + alpha * high_freq).clip(0, 1)
    return low_freq, high_freq, hybrid

def main():
    script_dir = os.path.dirname(__file__)
    img1_path = os.path.join(script_dir, "images", "dog.png")
    img2_path = os.path.join(script_dir, "images", "monkey.png")
    result_dir = os.path.join(script_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    img1 = read_image_gray32(img1_path)
    img2 = read_image_gray32(img2_path)

    # 兩張照片變成一樣的大小
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    low, high, hybrid = make_hybrid_image(img1, img2, ksize=15, sigma=5.0, alpha=1.0)

    save_image_gray(os.path.join(result_dir, "part2-2_low_freq.png"), low)
    save_image_gray(os.path.join(result_dir, "part2-2_high_freq.png"), (high + 0.5).clip(0, 1))
    save_image_gray(os.path.join(result_dir, "part2-2_hybrid.png"), hybrid)

if __name__ == "__main__":
    main()
