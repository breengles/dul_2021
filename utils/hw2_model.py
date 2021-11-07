import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm


class MaskedConv(nn.Conv2d):
    def __init__(self, dc=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.set_mask(dc)

    def _set_spatial_mask(self, isB=False):
        h, w = self.kernel_size
        self.mask[:, :, h // 2, : w // 2 + isB] = 1
        self.mask[:, :, : h // 2] = 1

    def _set_channel_mask(self, dc, isB=False):
        outc, inc, h, w = self.weight.shape
        self.mask[:, :, h // 2, w // 2] = self._mask_channel(inc, outc, dc, isB)

    def _mask_channel(self, inc, outc, dc, isB=False):
        inf = inc // dc + 1
        outf = outc // dc + 1

        mask = torch.tril(torch.ones((dc, dc)), isB - 1)

        # duplicating
        mask = torch.cat([mask] * inf, dim=1)
        mask = torch.cat([mask] * outf, dim=0)

        return mask[:outc, :inc]

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)  # applying mask
        return super().forward(x)  # call regular Conv2d

    def set_mask(self, dc, isB=False):
        raise NotImplementedError()


class ConvA(MaskedConv):
    def set_mask(self, dc):
        self._set_spatial_mask(False)
        self._set_channel_mask(dc, False)


class ConvB(MaskedConv):
    def set_mask(self, dc):
        self._set_spatial_mask(True)
        self._set_channel_mask(dc, True)


class SpatialConvA(MaskedConv):
    def set_mask(self, dc):
        self._set_spatial_mask(False)


class SpatialConvB(MaskedConv):
    def set_mask(self, dc):
        self._set_spatial_mask(True)


class ResidualBlock(nn.Module):
    def __init__(self, inc, color=True):
        super().__init__()
        h = inc // 2

        if color:
            self.main = nn.Sequential(
                ConvB(in_channels=inc, out_channels=h, kernel_size=1),
                nn.ReLU(),
                ConvB(in_channels=h, out_channels=h, kernel_size=7, padding=3),
                nn.ReLU(),
                ConvB(in_channels=h, out_channels=inc, kernel_size=1),
            )
        else:
            self.main = nn.Sequential(
                SpatialConvB(in_channels=inc, out_channels=h, kernel_size=1),
                nn.ReLU(),
                SpatialConvB(in_channels=h, out_channels=h, kernel_size=7, padding=3),
                nn.ReLU(),
                SpatialConvB(in_channels=h, out_channels=inc, kernel_size=1),
            )

    def forward(self, x):
        return self.main(x) + x


class IndependentPixelCNN(nn.Module):
    def __init__(self, input_shape, cf=120, num_classes=4):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.dc = c
        self.num_classes = num_classes

        self.model = nn.Sequential(
            SpatialConvA(in_channels=c, out_channels=cf, kernel_size=7, padding=3),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            ResidualBlock(cf, False),
            nn.ReLU(),
            SpatialConvB(in_channels=cf, out_channels=cf, kernel_size=1),
            nn.ReLU(),
            SpatialConvB(in_channels=cf, out_channels=c * num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        with torch.no_grad():
            outs = self(x).reshape(x.shape[0], self.num_classes, self.dc, *self.input_shape)
            proba = F.softmax(outs, dim=1)
            return proba

    @property
    def device(self):
        return self.model[0].weight.device

    def __step(self, batch):
        batch = batch.to(self.device)
        outs = self(batch).reshape(batch.shape[0], self.num_classes, self.dc, *self.input_shape)
        loss = F.cross_entropy(outs, batch.long())
        return loss

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
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self.__test(testloader))

        return self, losses

    def sample(self):
        with torch.no_grad():
            sample = torch.zeros(1, self.dc, *self.input_shape).to(self.device)

            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    probs = self.predict_proba(sample)[0, :, :, i, j].cpu().numpy()

                    labels = []
                    for c in range(self.dc):
                        labels.append(np.random.choice(self.num_classes, p=probs[:, c]))
                    labels = torch.tensor(labels, dtype=int)

                    sample[0, :, i, j] = labels

        return sample[0].cpu().numpy().transpose(1, 2, 0)


class PixelCNN(IndependentPixelCNN):
    def __init__(self, input_shape, cf=120, num_classes=4):
        super().__init__(input_shape, cf, num_classes)
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.dc = c
        self.num_classes = num_classes

        self.model = nn.Sequential(
            ConvA(in_channels=c, out_channels=cf, kernel_size=7, padding=3),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ResidualBlock(cf),
            nn.ReLU(),
            ConvB(in_channels=cf, out_channels=cf, kernel_size=1),
            nn.ReLU(),
            ConvB(in_channels=cf, out_channels=c * num_classes, kernel_size=1),
        )

    def sample(self):
        with torch.no_grad():
            sample = torch.zeros(1, self.dc, *self.input_shape).to(self.device)

            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    for c in range(self.dc):
                        probs = self.predict_proba(sample)[0, :, c, i, j].cpu().numpy()
                        sample[0, c, i, j] = torch.tensor(np.random.choice(self.num_classes, p=probs), dtype=int)

        return sample[0].cpu().numpy().transpose(1, 2, 0)
