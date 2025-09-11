import cv2, numpy as np, os

def read_rgb_float(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def save_image_rgb(path, img):
    cv2.imwrite(path,
                cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR))

# Gaussian stack
# 把原始照片用 filter 模糊，再把模糊的照片再模糊，共 level 次
def gaussian_stack(img, levels=8, ksize=15, sigma=2.0):
    stack = [img] # 先放入原始圖片
    for _ in range(1, levels):
        stack.append(cv2.GaussianBlur(stack[-1], (ksize, ksize), sigmaX=sigma))
    return stack

# Laplacian stack 是相鄰差
# 最後一張放上最後模糊的圖，因為沒東西可以減掉 
# 回傳的東西會像以下 [G0 - G1, G1 - G2, G2 - G3, G3 - G4] + [G4]
def laplacian_stack(g_stack):
    return [g_stack[i] - g_stack[i+1] for i in range(len(g_stack)-1)] + [g_stack[-1]]

def save_stack(stack, prefix, out_dir, is_lap=False):
    os.makedirs(out_dir, exist_ok=True)
    for i, layer in enumerate(stack):
        if is_lap and i != len(stack) - 1:
            # 為了讓 save 的照片看得到，所以才調
            # layer是Laplacian後的值，會有蠻多負數，如果直接用clip(0, 1)會幾乎都是黑色
            # 把他除以其中的最大值把值壓縮到[-1, 1]，再乘0.5和加0.5把他轉成[0, 1]
            layer_vis = layer / (np.max(np.abs(layer)) + 1e-8) * 0.5 + 0.5
        else:
            layer_vis = layer
        save_image_rgb(os.path.join(out_dir, f"{prefix}_level{i}.png"), layer_vis)

def main():
    root = os.path.dirname(__file__)
    img_dir, out_dir = os.path.join(root, "images"), os.path.join(root, "results")

    apple  = read_rgb_float(os.path.join(img_dir, "apple.jpeg"))
    orange = read_rgb_float(os.path.join(img_dir, "orange.jpeg"))

    g_apple = gaussian_stack(apple)
    l_apple = laplacian_stack(g_apple)

    g_orange = gaussian_stack(orange)
    l_orange = laplacian_stack(g_orange)

    save_stack(g_apple,  "part2-3_apple_gaussian",  out_dir)
    save_stack(l_apple,  "part2-3_apple_laplacian", out_dir, is_lap=True) # is_lap用true，因為要負值的問題

    save_stack(g_orange, "part2-3_orange_gaussian", out_dir)
    save_stack(l_orange, "part2-3_orange_laplacian", out_dir, is_lap=True)

if __name__ == "__main__":
    main()
