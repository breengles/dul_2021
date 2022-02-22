import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
from torch.distributions import Normal, Uniform
from scipy.optimize import bisect


class MaskedConv(nn.Conv2d):
    def __init__(self, color=True, dc=3, isB=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))

        self.color = color
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
    def __init__(self, input_shape, cf=120, num_gaussians=2, color=True):
        super().__init__()
        h, w, c = input_shape
        self.input_shape = (h, w)
        self.dc = c
        self.num_gaussians = num_gaussians

        self.model = nn.Sequential(
            MaskedConv(color=color, dc=c, isB=False, in_channels=c, out_channels=cf, kernel_size=7, padding=3),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            ResidualBlock(cf, color),
            MaskedConv(color=color, dc=c, isB=True, in_channels=cf, out_channels=cf, kernel_size=1),
            nn.ReLU(),
            MaskedConv(color=color, dc=c, isB=True, in_channels=cf, out_channels=c * num_gaussians * 3, kernel_size=1),
        )

    @property
    def bdist(self):
        return Uniform(torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

    def forward(self, x):
        return (
            self.model(x).reshape(x.shape[0], 3 * self.num_gaussians, self.dc, *self.input_shape).permute(0, 2, 1, 3, 4)
        )

    @property
    def device(self):
        return self.model[0].weight.device

    def loss(self, batch):
        z, log_det, *_ = self.flow(batch)
        log_prob = self.bdist.log_prob(z).to(self.device) + log_det
        return -log_prob.mean()

    def flow(self, batch):
        w_log, mu, log_s = self(batch).chunk(3, dim=2)
        w = F.softmax(w_log, dim=2)
        dist = Normal(mu, log_s.exp())

        x = batch.unsqueeze(1).repeat(1, 1, self.num_gaussians, 1, 1)
        z = (dist.cdf(x) * w).sum(2)
        log_det = (dist.log_prob(x).exp() * w).sum(2).log()

        return z, log_det, dist, w

    def _step(self, batch, train=True):
        batch = batch.to(self.device)

        if train:
            batch += Uniform(0.0, 0.5).sample(batch.shape).to(self.device)  # dequantizing

        return self.loss(batch)

    @torch.no_grad()
    def _test(self, testloader):
        losses = []

        for batch in tqdm(testloader, desc="Testing...", leave=False):
            losses.append(self._step(batch, False).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        # test before train
        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self._test(testloader))

        return self, losses

    def inverse(self, w, mu, log_s):
        n = w.shape[0]
        z = self.bdist.sample((n,))

        outs = []
        for i in range(n):
            dist = Normal(mu[i], log_s[i].exp())

            def closure(x):
                x = torch.FloatTensor(np.repeat(x, self.num_gaussians)).to(self.device)
                return (w[i] * dist.cdf(x)).sum() - z[i]

            outs.append(bisect(closure, -20, 20))

        return torch.FloatTensor(outs).to(self.device)

    @torch.no_grad()
    def sample(self, n=100):
        samples = torch.zeros(n, self.dc, *self.input_shape, device=self.device)

        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for c in range(self.dc):
                    w_log, mu, log_s = torch.chunk(self(samples), 3, dim=2)

                    # take pixel-channel specific values
                    w = F.softmax(w_log[:, c, :, i, j], dim=1)
                    mu = mu[:, c, :, i, j]
                    log_s = log_s[:, c, :, i, j]

                    samples[:, c, i, j] = self.inverse(w, mu, log_s)

        return samples.clip(0, 1).cpu().numpy().transpose(0, 2, 3, 1)


if __name__ == "__main__":
    batch = torch.ones((64, 1, 20, 20))

    model = PixelCNN((20, 20, 1), num_gaussians=15, color=False)

    model.sample(5)
