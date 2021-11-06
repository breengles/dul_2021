import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class OneHotDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        data = torch.tensor(data, dtype=int)
        xs_onehot = F.one_hot(data[:, 0])
        ys_onehot = F.one_hot(data[:, 1])

        self.data = torch.hstack((xs_onehot, ys_onehot)).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]  # (batch_size, 50)


class ImageOneHotDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = F.one_hot(torch.tensor(data, dtype=int)).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index].reshape(-1)


def get_dataloader(data, batch_size=64):
    dset = OneHotDataset(data)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


def get_img_dataloader(data, batch_size=64):
    dset = ImageOneHotDataset(data)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
