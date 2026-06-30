#!/usr/bin/env python3
import os
import struct
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# ==========================================
# 1. 数据读取与预处理模块 (保持不变)
# ==========================================
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def preprocess_images(images):
    num_images = images.shape[0]
    processed_images = np.zeros((num_images, 8, 8), dtype=np.float32)
    for i in range(num_images):
        resized = cv2.resize(images[i], (8, 8), interpolation=cv2.INTER_AREA)
        _, bw = cv2.threshold(resized, 35, 255, cv2.THRESH_BINARY)
        processed_images[i] = bw / 255.0
    return processed_images.reshape(num_images, 64)

# ==========================================
# 2. PyTorch 模型定义
# ==========================================
class MLP_64_16_10(nn.Module):
    def __init__(self):
        super(MLP_64_16_10, self).__init__()
        # 在硬件电路中，偏置项(bias)往往会增加加法器的开销。
        # 如果你想在 Logisim 中极简实现，可以将 bias=False，此处暂时保留默认 (bias=True)
        self.fc1 = nn.Linear(64, 16, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 
        return x

# ==========================================
# 3. 辅助函数：导出为 Logisim ROM 格式
# ==========================================
def export_to_logisim_hex(tensor, filename):
    """
    将 PyTorch Tensor 量化为 INT8 (补码形式) 并导出为 Logisim 可读的 HEX 文件
    Logisim 格式头部为: v2.0 raw
    """
    # 1. 获取最大绝对值以计算缩放因子
    max_val = torch.max(torch.abs(tensor)).item()
    scale = 127.0 / max_val if max_val != 0 else 1.0
    
    # 2. 量化：乘缩放因子 -> 四舍五入 -> 截断到 [-128, 127]
    quantized = torch.round(tensor * scale).clamp(-128, 127).to(torch.int8)
    
    # 3. 转换为 Logisim 兼容的十六进制字符串 (处理 8-bit 补码)
    # 负数在 8-bit 下的补码计算：val & 0xFF
    flat_data = quantized.flatten().tolist()
    hex_strings = [f"{val & 0xFF:02x}" for val in flat_data]
    
    # 4. 写入文件
    with open(filename, 'w') as f:
        f.write("v2.0 raw\n")
        # 每 16 个十六进制数为一行，方便阅读
        for i in range(0, len(hex_strings), 16):
            f.write(" ".join(hex_strings[i:i+16]) + "\n")
            
    print(f"已导出 {len(flat_data)} 个 INT8 权重至 {filename} (Scale factor: {scale:.4f})")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    # --- 1. 设备检查 (Apple Silicon MPS 加速) ---
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("已启用 Apple Silicon MPS 硬件加速!")
    else:
        device = torch.device("cpu")
        print("未检测到 MPS，使用 CPU 运行。")
    """
    device = torch.device("cpu")  # CPU 其实更快

    # --- 请在此处替换为你本地 MNIST 数据集的路径 ---
    train_images_path = 'dataset/train-images-idx3-ubyte'
    train_labels_path = 'dataset/train-labels-idx1-ubyte'
    val_images_path = 'dataset/t10k-images-idx3-ubyte'
    val_labels_path = 'dataset/t10k-labels-idx1-ubyte'
    
    print("正在读取和预处理数据...")
    X_train_raw = load_mnist_images(train_images_path)
    y_train_raw = load_mnist_labels(train_labels_path)
    X_val_raw = load_mnist_images(val_images_path)
    y_val_raw = load_mnist_labels(val_labels_path)

    X_train = preprocess_images(X_train_raw)
    X_val = preprocess_images(X_val_raw)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_raw, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val_raw, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    # 模型放到目标设备
    model = MLP_64_16_10().to(device)
    best_model = model.state_dict()
    best_val_acc = 0.0

    criterion = nn.CrossEntropyLoss() 
    
    epochs = 500

    # 【关键修改】：加入 weight_decay (L2 正则化)
    # 这会惩罚过大的权重值，使权重分布更加均匀，大幅降低 INT8 量化时的精度损失
    optimizer = optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n开始训练模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 数据放到目标设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                outputs = model(val_x)
                predictions = torch.argmax(outputs, dim=1)
                total += val_y.size(0)
                correct += (predictions == val_y).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%")

    # ==========================================
    # 5. 导出 Logisim INT8 权重
    # ==========================================
    print("\n--- 正在提取并量化权重用于 Logisim ---")
    # 将模型权重拉回 CPU 进行处理
    model.to("cpu") 
    
    # 提取第一层 (64x16) 和 第二层 (16x10) 的权重
    w1 = best_model['fc1.weight'].detach() # 形状: [16, 64]
    w2 = best_model['fc2.weight'].detach() # 形状: [10, 16]
    
    export_to_logisim_hex(w1, "logisim_fc1_weights.txt")
    export_to_logisim_hex(w2, "logisim_fc2_weights.txt")
    print("注意: Logisim 中使用 ROM 载入这些文件。电路中的乘法器需设置为 8-bit 2的补码(2's complement)模式。")

if __name__ == '__main__':
    main()