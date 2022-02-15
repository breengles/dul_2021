from copy import deepcopy
from turtle import forward

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms
from tqdm.auto import tqdm, trange


class Net(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, hid_dim_full=128):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(4)

        self.conv_to_fc = 4 * 6 * 6
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))

        self.features = nn.Linear(int(hid_dim_full // 2), out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = x.view(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        features = self.features(x)

        return features


class BYOL(nn.Module):
    def __init__(self, latent_dim=128) -> None:
        super().__init__()
        self.student = Net(1, latent_dim)
        self.teacher = deepcopy(self.student)
        self.teacher.requires_grad_(False)

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(24),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(9),
                transforms.Normalize(0.5, 0.5),
            ]
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    @torch.no_grad()
    def soft_update(source, target, tau=0.99):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * tp.data + (1 - tau) * sp.data)

    def _loss(self, v1, v2):
        z_student = self.student(v1)
        z_predictor = F.normalize(self.predictor(z_student))

        z_teacher = F.normalize(self.teacher(v2))

        return F.mse_loss(z_predictor, z_teacher, reduction="none").sum(-1).mean()

    def _step(self, batch):
        batch = batch.to(self.device)

        batch_student = self.transforms(batch)  # ==> v
        batch_teacher = self.transforms(batch)  # ==> v'

        loss_dir = self._loss(batch_student, batch_teacher)
        loss_exc = self._loss(batch_teacher, batch_student)

        return loss_dir + loss_exc

    def fit(self, trainloader, epochs=10, lr=1e-4):
        losses = []
        optim = Adam(self.student.parameters(), lr=lr)

        for _ in trange(epochs, desc="Training..."):
            for batch in trainloader:
                loss = self._step(batch[0])

                optim.zero_grad()
                loss.backward()
                optim.step()

                self.soft_update(self.student, self.teacher, 0.998)

                losses.append(loss.detach().cpu().numpy())

        return np.array(losses)

    @torch.no_grad()
    def encode(self, x):
        self.eval()

        x = x.to(self.device)
        x = transforms.Resize(24)(x)

        return self.student(x)
        # return self.predictor(self.student(x))


class BTWINS(nn.Module):
    def __init__(self, latent_dim=128, lam=1e-2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lam = lam
        self.main = Net(3, latent_dim)

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(28),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(9),
                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
                transforms.RandomGrayscale(0.2),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def forward(self, x):
        return self.main(x)

    @property
    def device(self):
        return next(self.parameters()).device


    @staticmethod
    def C(z1, z2):
        numerator = torch.einsum("bi,bj->ij", z1, z2)
        denom = torch.sqrt((z1 ** 2).sum(0)) * torch.sqrt((z2 ** 2).sum(0)).reshape(-1, 1)  # reshape to get ij matrix
        return numerator / denom / z1.shape[0]


    def step(self, batch):
        batch = batch.to(self.device)
        
        t1, t2 = self.transforms(batch), self.transforms(batch)
        z1, z2 = self(t1), self(t2)
        
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        
        c = self.C(z1, z2)
        invariance_term = ((1 - c.diag()) ** 2).sum()
        
        off_diag = c.masked_select(~torch.eye(self.latent_dim, dtype=bool))
        rr_term = self.lam * (off_diag ** 2).sum()
        
        loss = invariance_term + rr_term
        
        
        
        


    def fit(self, trainloader, epochs=10, lr=1e-4):
        losses = []

        optim = Adam(self.parameters(), lr=lr)        

        for _ in trange(epochs, desc="Training..."):
            for batch in trainloader:
                