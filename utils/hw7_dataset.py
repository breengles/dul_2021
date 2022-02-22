import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = (torch.tensor(data, dtype=torch.float32) - 0.5) * 2  # covert to [-1, 1] as G has tanh

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
