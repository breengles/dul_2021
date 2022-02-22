import torch
from torch.distributions import Uniform
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, test=False) -> None:
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))
        self.data /= 2

        if test:
            self.data += Uniform(0.0, 0.5).sample(self.data.shape)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]
