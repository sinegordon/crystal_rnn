import torch
from torch.utils.data import Dataset


class RNNCustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.as_tensor(x_data, dtype=torch.float32)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class RNNAutoEncoderCustomDataset(Dataset):
    def __init__(self, x_data):
        self.x_data = torch.as_tensor(x_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx]
