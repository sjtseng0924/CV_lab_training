import cv2
import numpy as np
import os

def read_image_rgb32(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2預設讀入是BGR，但其他函式都是RGB，所以要轉換  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img.astype(np.float32) / 255.0

def save_image_rgb(path, img):
    img_bgr = cv2.cvtColor((img * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

# sigma:3-6模糊圖片，1-2讓圖片平滑
# ksize:大約會是在6*sigma+1
def unsharp_mask_color(img, ksize=15, sigma=3.0, alpha=1.5):
    # 模糊後的img
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    # 原圖減去模糊後的圖，得到一些邊緣和細節
    high_freq = img - blurred
    # 將原圖和細節加權合併
    sharpened = (img + alpha * high_freq).clip(0, 1)
    return blurred, high_freq, sharpened

def main():
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "images", "taj.jpg")
    result_dir = os.path.join(script_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    img = read_image_rgb32(image_path)
    blurred, high, sharp = unsharp_mask_color(img, ksize=15, sigma=3.0, alpha=1.5)

    save_image_rgb(os.path.join(result_dir, "part2-1_blurred.png"), blurred)
    save_image_rgb(os.path.join(result_dir, "part2-1_high_freq.png"), (high + 0.5).clip(0, 1))  # shift for display
    save_image_rgb(os.path.join(result_dir, "part2-1_sharpened.png"), sharp)

if __name__ == "__main__":
    main()
