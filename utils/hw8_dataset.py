import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = 2 * torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32) - 1

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index]
