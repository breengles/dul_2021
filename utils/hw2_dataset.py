import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]


class ConditionalDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()

        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))
        self.labels = one_hot(torch.tensor(labels))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
