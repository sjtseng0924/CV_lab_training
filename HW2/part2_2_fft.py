import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_gray32(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def compute_spectrum(img):
    # fft2:2d傅立葉轉換，每個點是速度變化的方向，可看出高頻跟低頻的地方
    # 低頻:圖像中變化緩慢的部分，高頻：圖像中變化快速的部分
    f = np.fft.fft2(img)
    # 矩陣的兩個方向代表水平跟垂直的變換頻率高，左上角是低頻
    # fftshift:將低頻移到中心位置，變成四周是高頻
    fshift = np.fft.fftshift(f)
    # log1p，壓縮大數值，放大小數值
    magnitude = np.log1p(np.abs(fshift))  # log(1 + |F|)
    return magnitude

def plot_frequency_analysis(imgs, titles, save_path):
    plt.figure(figsize=(15, 8))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(compute_spectrum(img), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    script_dir = os.path.dirname(__file__)
    result_dir = os.path.join(script_dir, "results")

    dog_path = os.path.join(script_dir, "images", "dog.png")
    monkey_path = os.path.join(script_dir, "images", "monkey.png")
    low_path = os.path.join(result_dir, "part2-2_low_freq.png")
    high_path = os.path.join(result_dir, "part2-2_high_freq.png")
    hybrid_path = os.path.join(result_dir, "part2-2_hybrid.png")

    dog = read_gray32(dog_path)
    monkey = read_gray32(monkey_path)
    low = read_gray32(low_path)
    high = read_gray32(high_path)
    hybrid = read_gray32(hybrid_path)

    images = [dog, monkey, low, high, hybrid]
    titles = [
        "Dog Original", "Monkey Original",
        "Low-pass (Dog)", "High-pass (Monkey)",
        "Hybrid Image"
    ]

    plot_path = os.path.join(result_dir, "part2-2_frequency_analysis.png")
    plot_frequency_analysis(images, titles, plot_path)

if __name__ == "__main__":
    main()
