import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Uniform


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))
        self.data /= 2
        self.data += Uniform(0, 0.5).sample(self.data.shape)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]
