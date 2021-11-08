import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm


class MaskedConv(nn.Conv2d):
    def __init__(self, color=True, dc=3, isB=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))

        self.color = (color,)
        self.dc = dc
        self.isB = isB

        self.set_mask()

    def set_mask(self):
        outc, inc, h, w = self.weight.shape

        if self.color:
            self.mask[:, :, h // 2, : w // 2] = 1
            self.mask[:, :, : h // 2] = 1

            group_out, group_in = outc // self.dc, inc // self.dc

            if self.isB:
                self.mask[:group_out, :group_in, h // 2, w // 2] = 1
                self.mask[group_out : 2 * group_out, : 2 * group_in, h // 2, w // 2] = 1
                self.mask[2 * group_out :, :, h // 2, w // 2] = 1
            else:
                self.mask[group_out : 2 * group_out, :group_in, h // 2, w // 2] = 1
                self.mask[2 * group_out :, : 2 * group_in, h // 2, w // 2] = 1
        else:
            self.mask[:, :, h // 2, : w // 2 + self.isB] = 1
            self.mask[:, :, : h // 2] = 1

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)  # applying mask
        return super().forward(x)  # call regular Conv2d


class ResidualBlock(nn.Module):
    def __init__(self, inc, color=True):
        super().__init__()
        h = inc // 2

        self.main = nn.Sequential(
            nn.ReLU(),
            MaskedConv(color=color, dc=3, isB=True, in_channels=inc, out_channels=h, kernel_size=1),
            nn.ReLU(),
            MaskedConv(color=color, dc=3, isB=True, in_channels=h, out_channels=h, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv(color=color, dc=3, isB=True, in_channels=h, out_channels=inc, kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + x


class PixelCNN(nn.Module):
    def __init__(self, input_shape, cf=120, num_classes=4, color=True):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.dc = c
        self.num_classes = num_classes

        self.model = nn.Sequential(
            MaskedConv(color=color, dc=3, isB=False, in_channels=c, out_channels=cf, kernel_size=7, padding=3),
            nn.ReLU(),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            MaskedConv(color=color, dc=3, isB=True, in_channels=cf, out_channels=cf, kernel_size=1),
            nn.ReLU(),
            MaskedConv(color=color, dc=3, isB=True, in_channels=cf, out_channels=c * num_classes, kernel_size=1),
        )

    def forward(self, x):
        # return self.model(x).reshape(x.shape[0], self.num_classes, self.dc, *self.input_shape)
        return self.model(x).reshape(x.shape[0], self.dc, self.num_classes, *self.input_shape).permute(0, 2, 1, 3, 4)

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    @property
    def device(self):
        return self.model[0].weight.device

    def __step(self, batch):
        batch = batch.to(self.device)
        return F.cross_entropy(self(batch), batch.long())

    def __test(self, testloader):
        losses = []

        with torch.no_grad():
            for batch in tqdm(testloader, desc="Testing...", leave=False):
                losses.append(self.__step(batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        # test before train
        losses["test"].append(self.__test(testloader))

        for epoch in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                loss = self.__step(batch)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self.__test(testloader))

        return self, losses

    def sample(self, n):
        sample = torch.zeros(n, self.dc, *self.input_shape).to(self.device)
        with torch.no_grad():
            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    for c in range(self.dc):
                        probs = self.predict_proba(sample)[..., c, i, j]
                        sample[:, c, i, j] = torch.multinomial(probs, 1).flatten()

        return sample.cpu().numpy().transpose(0, 2, 3, 1)
