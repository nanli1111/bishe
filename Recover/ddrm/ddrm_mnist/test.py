# test_snr.py
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from ddrm.ddrm_mnist.dataset.dataset import get_dataloader, get_img_shape
from model.unet import UNet, build_network
from ddrm_core import DDRM

def add_gaussian_noise(imgs, snr_db):
    """根据 SNR 加噪"""
    # imgs 范围假设是 [-1,1]
    imgs_power = torch.mean(imgs**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = imgs_power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(imgs) * noise_std
    noisy_imgs = imgs + noise
    return noisy_imgs

def test_ddrm_snr(model, ddrm, dataloader, snr_list=[5,10,20,30], device='cuda', save_dir='ddrm_mnist/test_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    ddrm.model.eval()

    x_sample, _ = next(iter(dataloader))
    x_sample = x_sample[:8].to(device)  # 取前8张图像

    for snr_db in snr_list:
        x_noisy = add_gaussian_noise(x_sample, snr_db)
        x_denoised = ddrm.denoise(x_noisy)

        # 保存图像
        save_image((x_sample + 1)/2, os.path.join(save_dir, f'original.png'))
        save_image((x_noisy + 1)/2, os.path.join(save_dir, f'noisy_snr{snr_db}.png'))
        save_image((x_denoised + 1)/2, os.path.join(save_dir, f'denoised_snr{snr_db}.png'))

        print(f"SNR={snr_db} dB 完成去噪，并保存结果。")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== 配置模型 ========
    n_steps = 80
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load('ddrm_mnist/results/model_epoch50.pth', map_location=device))

    # DDRM 对象
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # 数据
    dataloader = get_dataloader(batch_size=16)

    # 测试不同 SNR
    snr_list = [-5, -10, -15, 0, 5]  # 单位 dB
    test_ddrm_snr(model, ddrm, dataloader, snr_list=snr_list, device=device)
