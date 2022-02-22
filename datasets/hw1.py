import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class OneHotDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        data = torch.tensor(data, dtype=int)

        xs_onehot = F.one_hot(data[:, 0])
        ys_onehot = F.one_hot(data[:, 1])

        self.data = torch.cat((xs_onehot.unsqueeze(1), ys_onehot.unsqueeze(1)), dim=1).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]  # (bs, 2, 25)


class ImageOneHotDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = F.one_hot(torch.tensor(data.reshape(data.shape[0], -1), dtype=int)).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]


def get_dataloader(data, batch_size=64):
    dset = OneHotDataset(data)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


def get_img_dataloader(data, batch_size=64):
    dset = ImageOneHotDataset(data)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


if __name__ == "__main__":
    import numpy as np

    data = np.random.randint(0, 25, size=(10000, 2))
    print(data.shape)

    dset = OneHotDataset(data)

    print(data[0])
    print(np.argmax(dset[0][:25]), np.argmax(dset[0][25:]))
