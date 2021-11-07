import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]


def get_dataloader(data, shuffle=False, batch_size=64):
    dset = MyDataset(data)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
