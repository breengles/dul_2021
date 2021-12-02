from torch.utils.data import Dataset
import torch
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = torch.tensor(data.transpose(0, 3, 1, 2), dtype=torch.float32) / 255 * 2 - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
