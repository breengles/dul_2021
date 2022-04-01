import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class CelebDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = data.astype(float)
        self.transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.transforms(self.data[index])
