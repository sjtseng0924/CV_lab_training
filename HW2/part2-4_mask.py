# 用自己畫定區域的照片當作mask
import cv2, os, numpy as np
import matplotlib.pyplot as plt

def read_rgb_float(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def gaussian_stack(img, levels=5, ksize=15, sigma=2.0):
    stack = [img]
    for _ in range(1, levels):
        stack.append(cv2.GaussianBlur(stack[-1], (ksize, ksize), sigmaX=sigma))
    return stack

def laplacian_stack(g_stack):
    return [g_stack[i] - g_stack[i+1] for i in range(len(g_stack)-1)] + [g_stack[-1]]

def blend_pyramid(lA, lB, g_mask):
    blended = [g_mask[i]*lA[i] + (1-g_mask[i])*lB[i] for i in range(len(lA))]
    img = blended[-1]
    for i in range(len(blended)-2, -1, -1):
        img += blended[i]
    return np.clip(img, 0, 1)

def save_rgb(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# 把自己想要當成mask的圖片讀進來
def load_mask_as_float(path, shape):
    # 讀成灰階的
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # 把mask的大小弄成跟其他圖片一樣
    resized = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    # 把gray mask疊三層讓他可以跟RGB圖片一起用
    return np.stack([resized] * 3, axis=2) 

def main():
    root = os.path.dirname(__file__)
    img_dir = os.path.join(root, "images")
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)

    car  = read_rgb_float(os.path.join(img_dir, "car.png"))
    face = read_rgb_float(os.path.join(img_dir, "face.png"))
    mask = load_mask_as_float(os.path.join(img_dir, "mask.png"), car.shape)

    h, w = car.shape[:2]
    face = cv2.resize(face, (w, h), cv2.INTER_AREA)

    gA = gaussian_stack(face)
    gB = gaussian_stack(car)
    lA = laplacian_stack(gA)
    lB = laplacian_stack(gB)
    # 一樣用不同模糊程度的mask
    g_mask = gaussian_stack(mask, levels=5, ksize=31, sigma=5.0)
    blended = blend_pyramid(lA, lB, g_mask)
    save_rgb(os.path.join(out_dir, "part2-4_blended.png"), blended)

if __name__ == "__main__":
    main()
