#!/usr/bin/env python3
import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols)

def preprocess_images(images):
    num_images = images.shape[0]
    processed_images = np.zeros((num_images, 8, 8), dtype=np.float32)
    
    for i in range(num_images):
        cropped = images[i][4:24, 4:24]
        resized = cv2.resize(cropped, (8, 8), interpolation=cv2.INTER_AREA)

        _, bw = cv2.threshold(resized, 115, 255, cv2.THRESH_BINARY)
        
        processed_images[i] = bw / 255.0
        
    return processed_images.reshape(num_images, 64)

# ==========================================
# 2. 可视化主程序
# ==========================================
def main():
    train_images_path = 'dataset/train-images-idx3-ubyte'
    
    print("正在读取数据集...")
    X_train_raw = load_mnist_images(train_images_path)
    
    # 提取前 20 张图片
    sample_raw = X_train_raw[:20]
    
    # 过一遍预处理管道
    sample_processed_1d = preprocess_images(sample_raw)
    
    # 把 (20, 64) 重新 reshape 成 (20, 8, 8) 以便画图
    sample_processed_2d = sample_processed_1d.reshape(20, 8, 8)
    
    # --- 开始绘图 (4行5列的网格) ---
    # figsize 控制弹窗的整体物理尺寸 (宽, 高)
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle("Hardware Input Map: 8x8 Binary Images (No Anti-Aliasing)", fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        # 【关键设置】：
        # cmap='gray': 黑白显示
        # interpolation='nearest': 强行关闭渲染抗锯齿，保留纯粹的像素方块！
        # vmin=0, vmax=1: 因为数据已经是 0.0 和 1.0 了，这能保证对比度最强
        ax.imshow(sample_processed_2d[i], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        
        # --- 附加外观优化：画出像素网格线 ---
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # 隐藏主坐标轴的刻度数字
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f"Index: {i}", fontsize=10)

    # 紧凑布局并弹出窗口
    plt.tight_layout()
    # 调整上边距，防止标题和图片重叠
    plt.subplots_adjust(top=0.92) 
    plt.show()

if __name__ == '__main__':
    main()