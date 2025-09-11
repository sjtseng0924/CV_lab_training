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

# 兩張圖片的 blend 的 mask
def create_smooth_mask(shape, width_ratio=0.3):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    # width_ratio 中間兩邊漸變的比例，佔整個寬度的0.3
    blend_w = int(w * width_ratio)
    start = w//2 - blend_w//2
    end   = w//2 + blend_w//2
    mask[:, :start] = 1.0 # 左邊用蘋果的圖
    mask[:, end:] = 0.0 # 右邊用橘子的圖
    # linespace 產生從1到0的等分的數值，並填入陣列中
    mask[:, start:end] = np.linspace(1, 0, end - start)
    # stack 用法: https://blog.csdn.net/qq_17550379/article/details/78934529
    # axis 是指哪一個維度要增加，原本是只有 3*3 這樣，因為增加第三維度，所以 1+1=2，整個矩陣變成3*3*2
    return np.stack([mask]*3, axis=2)  

def blend_pyramid(lA, lB, g_mask):
    # lA 和 lB 是 Laplacian stack，他們都是原圖 - 模糊圖的差值
    # 每個都用不同模糊程度的 mask 來混合
    blended = [g_mask[i]*lA[i] + (1-g_mask[i])*lB[i] for i in range(len(lA))]
    # 因為 laplacian stack 的最後一層是存模糊圖，所以由他當最開始加的那項
    img = blended[-1]
    # 把陣列 0 ~ len(la)-2 也加進去
    # 因為 laplacian stack 是差值，所以用最後的模糊圖加上這些差值，就可以還原回去
    for i in range(len(blended)-2, -1, -1):
        img += blended[i]
    return np.clip(img, 0, 1)

def save_rgb(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    root = os.path.dirname(__file__)
    img_dir = os.path.join(root, "images")
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)

    apple  = read_rgb_float(os.path.join(img_dir, "apple.jpeg"))
    orange = read_rgb_float(os.path.join(img_dir, "orange.jpeg"))
    h, w = apple.shape[:2]

    orange = cv2.resize(orange, (w, h), cv2.INTER_AREA)
    gA, gB = gaussian_stack(apple), gaussian_stack(orange)
    lA, lB = laplacian_stack(gA), laplacian_stack(gB)

    smooth_mask = create_smooth_mask((h, w))
    # mask 的模糊程度也是不同層次的 對於每一個頻率的照片 用不同的mask去混和
    g_mask = gaussian_stack(smooth_mask)

    # Blend
    oraple = blend_pyramid(lA, lB, g_mask)
    save_rgb(os.path.join(out_dir, "part2-4_oraple.png"), oraple)

    # 印出mask長甚麼樣
    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1); plt.imshow(apple); plt.title("Apple"); plt.axis('off')
    plt.subplot(2,2,2); plt.imshow(smooth_mask); plt.title("Mask"); plt.axis('off')
    plt.subplot(2,2,3); plt.imshow(orange); plt.title("Orange"); plt.axis('off')
    plt.subplot(2,2,4); plt.imshow(oraple); plt.title("Oraple"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part2-4_oraple_summary.png"))
    plt.close()

if __name__ == "__main__":
    main()
