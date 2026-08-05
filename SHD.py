import torch

import h5py
import numpy as np



class SHD(torch.utils.data.Dataset):
    def __init__(self, split, num_steps=300, t_max=1.4):

        file_path = f"./data/SHD/shd_{split}.h5"

        self.file = h5py.File(file_path, "r")

        self.times = self.file["spikes"]["times"]

        self.units = self.file["spikes"]["units"]
        self.labels = self.file["labels"]

        self.num_steps = num_steps
        self.n_inputs = 700
        self.dt = t_max / num_steps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        times = np.asarray(self.times[idx])
        units = np.asarray(self.units[idx])

        spikes = torch.zeros(self.num_steps, self.n_inputs)

        bins = (times / self.dt).astype(np.int64)

        valid = (bins >= 0) & (bins < self.num_steps)

        spikes[bins[valid], units[valid]] = 1.0

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spikes, label