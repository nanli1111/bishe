import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class QPSKDataset(Dataset):
    def __init__(self,
                 start_samples: int,
                 end_samples: int,
                 data_dir: r'F:\LJN\bishe\bishe\Estimate\data\tdl_data_5h',
                 expand_h_to_wave: bool = False):
        """
        data_dir 里需要有：
          impaired_waveforms.npy  (N,2,176)
          clean_waveforms.npy     (N,2,176)
          labels.npy              (N,)
          true_h.npy              (N,10,11)
          estimated_h.npy         (N,10,11)

        expand_h_to_wave:
          False -> 返回 (10,11)
          True  -> 返回 (10,176)  (11*16=176)
        """
        self.expand_h_to_wave = expand_h_to_wave

        self.impaired = np.load(os.path.join(data_dir, "impaired_waveforms.npy"), mmap_mode="r")
        self.clean    = np.load(os.path.join(data_dir, "clean_waveforms.npy"), mmap_mode="r")
        self.labels   = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode="r")
        self.true_h   = np.load(os.path.join(data_dir, "true_h.npy"), mmap_mode="r")
        self.est_h    = np.load(os.path.join(data_dir, "estimated_h.npy"), mmap_mode="r")

        # slice
        self.impaired = self.impaired[start_samples:end_samples]
        self.clean    = self.clean[start_samples:end_samples]
        self.labels   = self.labels[start_samples:end_samples]
        self.true_h   = self.true_h[start_samples:end_samples]
        self.est_h    = self.est_h[start_samples:end_samples]

        # 固定从你的 shape 推出来
        self.L = self.clean.shape[2]      # 176
        self.S = self.est_h.shape[2]      # 11
        assert self.L % self.S == 0, f"L({self.L}) 不能整除 S({self.S})"
        self.sps = self.L // self.S       # 16

    def __len__(self):
        return self.clean.shape[0]

    def _expand_h(self, h_sym):
        """(10,11) -> (10,176)"""
        return np.repeat(h_sym, repeats=self.sps, axis=-1)

    def __getitem__(self, idx):
        clean = self.clean[idx].astype(np.float32)        # (2,176)
        impaired = self.impaired[idx].astype(np.float32)  # (2,176)
        label = int(self.labels[idx])                     # scalar
        true_h = self.true_h[idx].astype(np.float32)      # (10,11)
        est_h  = self.est_h[idx].astype(np.float32)       # (10,11)

        if self.expand_h_to_wave:
            true_h = self._expand_h(true_h)               # (10,176)
            est_h  = self._expand_h(est_h)                # (10,176)

        # 转 torch
        return (
            torch.from_numpy(clean),
            torch.from_numpy(impaired),
            torch.from_numpy(est_h),
            torch.from_numpy(true_h),
            torch.tensor(label, dtype=torch.long),
        )


def get_train_loader(data_dir, start, end, batch_size=64, val_split=0.2,
                     expand_h_to_wave=False, shuffle=True):
    ds = QPSKDataset(start, end, data_dir=data_dir, expand_h_to_wave=expand_h_to_wave)
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_test_loader(data_dir, start, end, batch_size=64, expand_h_to_wave=False):
    ds = QPSKDataset(start, end, data_dir=data_dir, expand_h_to_wave=expand_h_to_wave)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_signal_shape(data_dir):
    clean = np.load(os.path.join(data_dir, "clean_waveforms.npy"), mmap_mode="r")
    return (clean.shape[1], clean.shape[2])  # (2,176)
