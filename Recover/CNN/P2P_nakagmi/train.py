import os
import math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# 项目模块
from dataset.dataset import QPSKDataset
from model.resnet import RESNET_model

# 复用 test.py 里的加噪函数（不要再自己写）
from test_ber import add_awgn_noise_np


# ==========================
#  标签 0/1/2/3 → I/Q 两路 bit
# ==========================
def create_targets(y, device):
    """
    y: [B]，符号标签 0,1,2,3

    映射规则（与你原来的程序一致）：
      0 -> (I=-1, Q=-1)
      1 -> (I=-1, Q=+1)
      2 -> (I=+1, Q=+1)
      3 -> (I=+1, Q=-1)

    这里我们用 0/1 表示 I/Q 两路 bit，方便配合 BCEWithLogitsLoss：
      I 分支 (y0):
        符号 2,3 为 +1 -> 0
        符号 0,1 为 -1 -> 1

      Q 分支 (y1):
        符号 1,2 为 +1 -> 0
        符号 0,3 为 -1 -> 1
    """
    y = y.to(device)
    y0 = torch.zeros_like(y, dtype=torch.float32, device=device)
    y1 = torch.zeros_like(y, dtype=torch.float32, device=device)

    mask_0 = (y == 0)
    mask_1 = (y == 1)
    mask_2 = (y == 2)
    mask_3 = (y == 3)

    # I 分支
    y0[mask_1 | mask_2] = 0
    y0[mask_0 | mask_3] = 1

    # Q 分支
    y1[mask_2 | mask_3] = 0
    y1[mask_0 | mask_1] = 1

    return y0, y1


# ==========================
#        训练函数
# ==========================
def train_resnet(model,
                 train_loader,
                 val_loader,
                 epochs=100,
                 lr=1e-3,
                 weight_decay=1e-4,
                 device='cuda',
                 patience=8,
                 checkpoint_path='./best_resnet_model.pth',
                 save_dir='./results'):

    os.makedirs(save_dir, exist_ok=True)

    # 两个分支各一个损失函数（I/Q）
    criterion_i = nn.BCEWithLogitsLoss()
    criterion_q = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    early_stop_counter = 0

    print("开始训练 ResNet 模型 ...")

    for epoch in range(1, epochs + 1):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=110)

        for batch in train_pbar:
            # Dataset __getitem__ 返回：x, y, z, labels
            x_clean, y_impaired, z_h, labels = batch

            # 按你原来的接口：model(x, z)
            # 这里把“受损信号 y_impaired”作为输入信号
            x = x_impaired = y_impaired.float().to(device)
            z = z_h.float().to(device)
            y = labels.long().to(device)

            optimizer.zero_grad()

            outputs = model(x, z)         # [B, 2]
            outputs_i = outputs[:, 0]
            outputs_q = outputs[:, 1]

            # 创建 I/Q 目标 bits
            y0, y1 = create_targets(y, device)  # [B], [B]

            loss0 = criterion_i(outputs_i, y0)
            loss1 = criterion_q(outputs_q, y1)
            loss = loss0 + loss1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        correct0, correct1, total = 0, 0, 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", ncols=110)

        with torch.no_grad():
            for batch in val_pbar:
                x_clean, y_impaired, z_h, labels = batch

                x = y_impaired.float().to(device)
                z = z_h.float().to(device)
                y = labels.long().to(device)

                outputs = model(x, z)
                outputs_i = outputs[:, 0]
                outputs_q = outputs[:, 1]

                y0, y1 = create_targets(y, device)

                loss0 = criterion_i(outputs_i, y0)
                loss1 = criterion_q(outputs_q, y1)
                loss = loss0 + loss1
                val_loss += loss.item()

                # 二分类：阈值 0 → 0/1
                predicted0 = (outputs_i > 0).float()
                predicted1 = (outputs_q > 0).float()

                correct0 += (predicted0 == y0).sum().item()
                correct1 += (predicted1 == y1).sum().item()
                total += y.size(0)

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        acc = (correct0 + correct1) / (2 * total)

        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.6f}, Val Acc: {acc:.4f}")

        # ---------- early stopping & checkpoint ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ 验证损失下降，保存模型到 {checkpoint_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⚠️ 验证损失没有下降 ({early_stop_counter}/{patience})")

            if early_stop_counter >= patience:
                print("⏹️ 早停触发，停止训练。")
                break

    # 画一下 loss 曲线（风格跟你 IS2B 模板一致）
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'resnet_loss.png'))

    return train_losses, val_losses


# ==========================
#            主函数
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ------- 超参数、路径、接口层东西都放这里 -------
    batch_size = 64
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    patience = 8    

    checkpoint_path = r"F:\LJN\bishe\bishe\CNN\P2P_nakagmi\results\best_resnet_model.pth"
    save_dir = r"F:\LJN\bishe\bishe\CNN\P2P_nakagmi\results"

    # 1. 准备数据集（参照你给的模板）
    train_data = QPSKDataset(0, 400000)

    # 2. 对 y_impaired 加噪声，SNR = 10 - 10 * log_10(16)
    snr_db = 10 - 10 * math.log(16, 10) + 10 * math.log10(2)   # 这里用 math.log(x, 10) 明确以 10 为底
    # 注意：不做类型转换，保持和 test.py 里的 add_awgn_noise 用法一致
    train_data.y = add_awgn_noise_np(train_data.y, snr_db)

    # 3. 划分训练集 / 验证集（80% / 20%）
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 4. 构建模型
    model = RESNET_model(num_classes=2, dropout=0.2, use_se=True).to(device)

    # 5. 开始训练
    train_losses, val_losses = train_resnet(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        patience=patience,
        checkpoint_path=checkpoint_path,
        save_dir=save_dir,
    )
