import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

from .base import BaseModule


class VAT(BaseModule):
    def __init__(self, xi=1.0):
        super().__init__()

        self.xi = xi

        self.conv = nn.Sequential(
            # conv96
            nn.Conv2d(3, 96, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(96),
            # first maxpool
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(),
            # first conv 192
            nn.Conv2d(96, 192, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(192),
            # second maxpool
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(),
            # second conv 192
            nn.Conv2d(192, 192, (3, 3)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(192, 192, (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(192, 192, (1, 1)),
            nn.LeakyReLU(0.1),
        )

        self.clf = nn.Sequential(nn.Flatten(), nn.Linear(192, 10))

    def adv_loss(self, x):
        with torch.no_grad():
            pred = F.softmax(self(x), dim=1)  # literally p(y|x)

        r = self.xi * torch.randn(x.shape).to(self.device).requires_grad_()

        pred_hat = self(x + r)
        logp_hat = F.log_softmax(pred_hat, dim=1)  # log p(y|x + r, theta)
        distribution_distance = F.kl_div(logp_hat, pred, reduction="batchmean")

        r_vadv = F.normalize(torch.autograd.grad(distribution_distance, r)[0], dim=(1, 2, 3))
        self.zero_grad()

        pred_hat = self(x + r_vadv)  # log p(y|x + r_vadv, theta)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction="batchmean")
        return loss

    def forward(self, x):
        return self.clf(self.conv(x))

    @torch.no_grad()
    def test(self, testloader):
        self.eval()

        acc = 0
        for imgs, labels in testloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = torch.argmax(F.softmax(self(imgs), dim=1), dim=1)

            acc += (preds == labels).sum().float()

        acc /= len(testloader.dataset)

        self.train()

        return acc.cpu().numpy()

    def step(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        labeled = labels != -1

        adv_loss = self.adv_loss(imgs)

        if labeled.sum() == 0:
            clf_loss = 0
        else:
            clf_loss = F.cross_entropy(self(imgs[labeled]), labels[labeled])

        return clf_loss + adv_loss


class FixMatch(BaseModule):
    def __init__(self, out_dim=128, hid_dim_full=128, tau=0.7, lam=10):
        super().__init__()
        self.tau = tau
        self.lam = lam

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)

        self.conv_to_fc = 8 * 8 * 4
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))

        self.features = nn.Linear(int(hid_dim_full // 2), out_dim)
        self.last = nn.Linear(out_dim, 10)

        self.weak_transform = T.Compose([T.RandomHorizontalFlip(), T.Normalize((0.5,), (0.5,))])

        self.strong_tansform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=32),
                T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=9),
                T.Normalize((0.5,), (0.5,)),
            ]
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.reshape(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        features = self.features(x)

        return self.last(features)

    @torch.no_grad()
    def test(self, testloader):
        self.eval()

        acc = 0
        for imgs, labels in testloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = torch.argmax(F.softmax(self(imgs), dim=1), dim=1)

            acc += (preds == labels).sum().float()

        acc /= len(testloader.dataset)

        self.train()

        return acc.cpu().numpy()

    def step(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        labeled = labels != -1
        unlabeled = labels == -1

        if labeled.sum() == 0:
            loss_labeled = 0
        else:
            pred_labeled = self(self.weak_transform(imgs[labeled]))
            loss_labeled = F.cross_entropy(pred_labeled, labels[labeled])

        if unlabeled.sum() == 0:
            loss_unlabeled = 0
        else:
            pred_unlabeled = self(self.weak_transform(imgs[unlabeled]))

            confidence, pseudolabels = torch.max(pred_unlabeled, dim=1)
            thresholded = confidence > self.tau

            if thresholded.sum() == 0:
                loss_unlabeled = 0
            else:
                loss_unlabeled = F.cross_entropy(
                    self(self.strong_tansform(imgs[unlabeled][thresholded])), pseudolabels[thresholded]
                )

        return loss_labeled + self.lam * loss_unlabeled
