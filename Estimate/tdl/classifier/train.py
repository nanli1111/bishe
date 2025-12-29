import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import math

from dataset import get_dataloader
from model import OptimizedQPSKNet
from noise_utils import add_awgn_noise_torch

SAVE_DIR = r'F:\LJN\bishe\bishe\Estimate\tdl\classifier\results'

CONFIG = {
    'train_samples': 400000,
    'val_split': 0.2,
    'batch_size': 128,
    'epochs': 200,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': os.path.join(SAVE_DIR, 'best_qpsk_model.pth'),
    'patience': 10,
    'snr_min': 0,
    'snr_max': 20,
    'val_snr': 10,
    'sps': 16
}

# --------------------------------------------------
# label -> Gray bits (I/Q): 0->00, 1->10, 2->11, 3->01
# bit=1 表示负号；bit顺序：[I_bit, Q_bit]
# --------------------------------------------------
def labels_to_bits_gray_iq(label, device):
    label = label.to(device).long()
    lut = torch.tensor([
        [0.0, 0.0],  # 0: + +
        [1.0, 0.0],  # 1: - +
        [1.0, 1.0],  # 2: - -
        [0.0, 1.0],  # 3: + -
    ], device=device)
    return lut[label]  # [B,2] float(0/1)

def compute_bit_errors_from_logits(logits, targets_bits):
    """
    logits: [B,2] (I/Q logits)
    targets_bits: [B,2] float(0/1)
    """
    pred_bits = (logits > 0).to(targets_bits.dtype)      # [B,2] float(0/1)
    bit_errors = (pred_bits != targets_bits).sum().item()
    total_bits = targets_bits.numel()
    return bit_errors, total_bits

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Starting Training on {CONFIG['device']}...")

    val_count = int(CONFIG['train_samples'] * CONFIG['val_split'])
    train_count = CONFIG['train_samples'] - val_count

    train_loader = get_dataloader(0, train_count, CONFIG['batch_size'], shuffle=True)
    val_loader = get_dataloader(train_count, CONFIG['train_samples'], CONFIG['batch_size'], shuffle=False)

    # 输出 2 个 logit（I/Q 两个 bit）
    model = OptimizedQPSKNet().to(CONFIG['device'])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    best_val_ber = float("inf")
    history = {'train_loss': [], 'val_loss': [], 'val_ber': []}
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        # ---------- Train ----------
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", ncols=110)
        for imp, h, label in pbar:
            imp = imp.to(CONFIG['device']).float()
            h = h.to(CONFIG['device']).float()
            label = label.to(CONFIG['device']).long()

            # 动态加噪：每个 batch 一个 SNR
            snr_symbol = np.random.uniform(CONFIG['snr_min'], CONFIG['snr_max'])
            snr_db_sample = snr_symbol - 10 * math.log10(CONFIG['sps']) + 10 * math.log10(2)
            noisy_imp = add_awgn_noise_torch(imp, snr_db_sample)

            targets = labels_to_bits_gray_iq(label, CONFIG['device'])  # [B,2] float

            optimizer.zero_grad()
            logits = model(noisy_imp, h)  # [B,2]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'snr': f"{snr_symbol:.1f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ---------- Val (算 val_loss + val_ber) ----------
        model.eval()
        running_val_loss = 0.0
        val_bit_errors = 0
        val_total_bits = 0

        with torch.no_grad():
            for imp, h, label in val_loader:
                imp = imp.to(CONFIG['device']).float()
                h = h.to(CONFIG['device']).float()
                label = label.to(CONFIG['device']).long()

                snr_symbol = CONFIG['val_snr']
                snr_db_sample = snr_symbol - 10 * math.log10(CONFIG['sps']) + 10 * math.log10(2)
                noisy_imp = add_awgn_noise_torch(imp, snr_db_sample)

                targets = labels_to_bits_gray_iq(label, CONFIG['device'])  # [B,2]
                logits = model(noisy_imp, h)  # [B,2]

                loss = criterion(logits, targets)
                running_val_loss += loss.item()

                be, tb = compute_bit_errors_from_logits(logits, targets)
                val_bit_errors += be
                val_total_bits += tb

        avg_val_loss = running_val_loss / len(val_loader)
        val_ber = val_bit_errors / max(1, val_total_bits)

        history['val_loss'].append(avg_val_loss)
        history['val_ber'].append(val_ber)

        print(f"Epoch {epoch+1} Result: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val BER={val_ber:.3e} (SNR={CONFIG['val_snr']} dB)")

        # ---------- early stopping & checkpoint（按 val_ber 越小越好） ----------
        if val_ber < best_val_ber:
            best_val_ber = val_ber
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"  --> Model saved with Val BER {best_val_ber:.3e}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG['patience']:
                print(f"\nEARLY STOPPING. Best Val BER: {best_val_ber:.3e}")
                break

    # ---------- Plot ----------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss', linestyle='--')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(history['val_ber'], label='Val BER')
    plt.legend()
    plt.grid(True, which='both')

    fig_path = os.path.join(SAVE_DIR, 'training_history.png')
    plt.savefig(fig_path, dpi=200)
    print(f"Training finished. History saved to {fig_path}")

if __name__ == "__main__":
    train()
