import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math

from dataset import get_dataloader
from model import OptimizedQPSKNet
from noise_utils import add_awgn_noise_torch

SAVE_DIR = r'F:\LJN\bishe\bishe\Estimate\tdl\classifier\results'

CONFIG = {
    'test_start': 400000,
    'test_end': 490000,
    'batch_size': 128,
    'model_path': os.path.join(SAVE_DIR, 'best_qpsk_model.pth'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'snr_list': np.arange(0, 20, 1),
    'sps': 16
}

# label(0/1/2/3) -> Gray bits (I/Q): 0->00, 1->10, 2->11, 3->01
def labels_to_bits_gray_iq(label, device):
    label = label.to(device).long()
    lut = torch.tensor([
        [0, 0],  # 0: + +
        [1, 0],  # 1: - +
        [1, 1],  # 2: - -
        [0, 1],  # 3: + -
    ], dtype=torch.int64, device=device)
    return lut[label]  # [B,2]

def test():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: Model not found: {CONFIG['model_path']}")
        return

    print(f"Starting SNR Sweep Testing on {CONFIG['device']}...")

    test_loader = get_dataloader(CONFIG['test_start'], CONFIG['test_end'],
                                 CONFIG['batch_size'], shuffle=False)

    # 输出 2 个 logit（I/Q 两个 bit）
    model = OptimizedQPSKNet(num_classes=2).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    model.eval()

    bers = []

    print("\nSNR (dB) | BER")
    print("---------|----------------")

    for snr in CONFIG['snr_list']:
        bit_errors = 0
        total_bits = 0

        with torch.no_grad():
            for imp, h, label in tqdm(test_loader, desc=f"SNR={snr:.1f}", ncols=110, leave=False):
                imp = imp.to(CONFIG['device']).float()
                h = h.to(CONFIG['device']).float()
                label = label.to(CONFIG['device']).long()

                snr_db_sample = snr - 10 * math.log10(CONFIG['sps']) + 10 * math.log10(2)
                noisy_imp = add_awgn_noise_torch(imp, snr_db_sample)

                logits = model(noisy_imp, h)           # [B,2]
                pred_bits = (logits > 0).long()        # [B,2] 0/1

                true_bits = labels_to_bits_gray_iq(label, CONFIG['device'])  # [B,2]

                bit_errors += (pred_bits != true_bits).sum().item()
                total_bits += label.size(0) * 2

        ber = bit_errors / total_bits
        bers.append(ber)
        print(f"{snr:>8.1f} | {ber:.6e}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogy(CONFIG['snr_list'], bers, 'b-o', linewidth=2, label='Proposed (2-bit BCE)')

    plt.title('BER vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    save_file = os.path.join(SAVE_DIR, "snr_ber_curve.png")
    plt.savefig(save_file, dpi=300)
    print(f"\nResult curve saved to {save_file}")

    # CSV（控制浮点位数）
    csv_path = os.path.join(SAVE_DIR, "snr_ber_results.csv")
    np.savetxt(
        csv_path,
        np.column_stack((CONFIG['snr_list'], bers)),
        delimiter=",",
        header="SNR,BER",
        comments="",
        fmt=["%.1f", "%.3e"]
    )
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    test()
