import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate


class RDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()

        self.data = data
        self.rotations = [0, 90, 180, 270]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        r = np.random.choice(len(self.rotations))
        return rotate(self.data[index][0], self.rotations[r]), r
