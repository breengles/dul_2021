from ast import Not
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal, Uniform
from torch.nn import functional as F
import numpy as np
from tqdm.auto import tqdm, trange


class Conv2d_wn(nn.Module):
    """
    just simple weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.main = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.main(x)


class ActNorm(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.n_channels = n_channels

        self.log_s = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)

        self.initialized = False

    @property
    def device(self):
        return next(self.parameters()).device

    def __initialize_weights(self, x):
        self.b.data = -torch.mean(x, dim=(0, 2, 3), keepdim=True)
        s = torch.std(x.permute(1, 0, 2, 3).reshape(self.n_channels, -1), dim=1).reshape(1, self.n_channels, 1, 1)
        self.log_s.data = -torch.log(s)

        self.initialized = True

    def forward(self, x, reverse=False):
        if reverse:
            return (x - self.b) * torch.exp(-self.log_s), self.log_s

        if not self.initialized:
            self.__initialize_weights(x)

        return x * self.log_s.exp() + self.b, self.log_s


class ResBlock(nn.Module):
    """
    h = x
    h = conv2d(n_filters, n_filters, (1,1), stride=1, padding=0)(h)
    h = relu(h)
    h = conv2d(n_filters, n_filters, (3,3), stride=1, padding=1)(h)
    h = relu(h)
    h = conv2d(n_filters, n_filters, (1,1), stride=1, padding=0)(h)
    return h + x
    """

    def __init__(self, n_filters) -> None:
        super().__init__()

        self.main = nn.Sequential(
            Conv2d_wn(n_filters, n_filters, (1, 1), 1, 0),
            nn.ReLU(),
            Conv2d_wn(n_filters, n_filters, (3, 3), 1, 1),
            nn.ReLU(),
            Conv2d_wn(n_filters, n_filters, (1, 1), 1, 0),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return x + self.main(x)


class SimpleResnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=128) -> None:
        super().__init__()

        self.main = nn.Sequential(
            Conv2d_wn(in_channels, n_filters, (3, 3), 1, 1),
            ResBlock(n_filters),
            ResBlock(n_filters),
            ResBlock(n_filters),
            ResBlock(n_filters),
            # ResBlock(n_filters),
            # ResBlock(n_filters),
            # ResBlock(n_filters),
            # ResBlock(n_filters),
            nn.ReLU(),
            Conv2d_wn(n_filters, out_channels, (3, 3), 1, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.main(x)


class AffineCouplingWithCheckerboard(nn.Module):
    def __init__(self, top_left=1) -> None:
        super().__init__()

        self.mask = self.__build_mask(top_left)

        self.g = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.resnet = SimpleResnet(3, 6)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def __build_mask(top_left=1):
        mask = torch.arange(32).reshape(-1, 1) + torch.arange(32)  # 32x32 board

        # top-left corner is 1 as (1 + 1 + 1) % 2 = 1 if top_left == 1
        # see fig 3 (left) in https://arxiv.org/pdf/1605.08803.pdf
        mask = torch.remainder(top_left + mask, 2)

        mask = mask.reshape(-1, 1, 32, 32)
        return mask.float()

    def forward(self, x, reverse=False):
        bs, n_channels, *_ = x.shape
        mask = self.mask.repeat(bs, 1, 1, 1).to(x.device)

        x_ = x * mask

        log_s, b = self.resnet(x_).split(n_channels, dim=1)
        log_s = self.g * torch.tanh(log_s) + self.b

        b = b * (1 - mask)
        log_s = log_s * (1 - mask)

        if reverse:
            x = (x - b) * torch.exp(-log_s)
        else:
            x = x * log_s.exp() + b

        return x, log_s


class AffineCouplingWithChannels(nn.Module):
    def __init__(self, topleft) -> None:
        super().__init__()
        self.topleft = topleft

        self.g = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.resnet = SimpleResnet(6, 12)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, reverse=False):
        bs, n_channels, *_ = x.shape

        if self.topleft:
            on, off = x.split(n_channels // 2, dim=1)
        else:
            off, on = x.split(n_channels // 2, dim=1)

        log_s, b = self.resnet(off).split(n_channels // 2, dim=1)
        log_s = self.g * torch.tanh(log_s) + self.b

        if reverse:
            on = (on - b) * torch.exp(-log_s)
        else:
            on = on * log_s.exp() + b

        if self.topleft:
            return torch.cat([on, off], dim=1), torch.cat([log_s, torch.zeros_like(log_s)], dim=1)
        else:
            return torch.cat([off, on], dim=1), torch.cat([torch.zeros_like(log_s), log_s], dim=1)


class RealNVP(nn.Module):
    def __init__(self, alpha=0.05) -> None:
        super().__init__()
        self.alpha = alpha

        self.bdist = None  # will be initialized in `fit`

        self.checker_tranforms = [
            nn.ModuleList(
                [
                    AffineCouplingWithCheckerboard(1),
                    ActNorm(3),
                    AffineCouplingWithCheckerboard(0),
                    ActNorm(3),
                    AffineCouplingWithCheckerboard(1),
                    ActNorm(3),
                    AffineCouplingWithCheckerboard(0),
                ]
            ),
            nn.ModuleList(
                [
                    AffineCouplingWithCheckerboard(1),
                    ActNorm(3),
                    AffineCouplingWithCheckerboard(0),
                    ActNorm(3),
                    AffineCouplingWithCheckerboard(1),
                ]
            ),
        ]

        self.channel_tranforms = nn.ModuleList(
            [
                AffineCouplingWithChannels(True),
                ActNorm(12),
                AffineCouplingWithChannels(False),
                ActNorm(12),
                AffineCouplingWithChannels(True),
            ]
        )

    @staticmethod
    def squeeze(x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c * 4, h // 2, w // 2)
        return x

    @staticmethod
    def unsqueeze(x):
        b, c, h, w = x.shape
        x = x.reshape(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c // 4, h * 2, w * 2)
        return x

    def g(self, z):
        x = z

        for t in reversed(self.checker_tranforms[1]):
            x, _ = t(x, True)

        x = self.squeeze(x)

        for t in reversed(self.channel_tranforms):
            x, _ = t(x, True)

        x = self.unsqueeze(x)

        for t in reversed(self.checker_tranforms[0]):
            x, _ = t(x, True)

        return x

    def f(self, x):
        z, log_det = x, torch.zeros_like(x)

        for t in self.checker_tranforms[0]:
            z, d = t(z)
            log_det += d

        z, log_det = self.squeeze(z), self.squeeze(log_det)

        for t in self.channel_tranforms:
            z, d = t(z)
            log_det += d

        z, log_det = self.unsqueeze(z), self.unsqueeze(z)

        for t in self.checker_tranforms[1]:
            z, d = t(z)
            log_det += d

        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        p_x = log_det.sum(dim=(1, 2, 3))
        p_z = self.bdist.log_prob(z).sum(dim=(1, 2, 3))
        return p_x + p_z

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        for t in self.channel_tranforms:
            t.to(*args, **kwargs)

        for ts in self.checker_tranforms:
            for t in ts:
                t.to(*args, **kwargs)

        return self

    def preprocess(self, x):
        x += Uniform(0, 0.1).sample(x.shape).to(self.device)
        x = x.clip(0, 1)

        x *= 1 - self.alpha
        x += self.alpha

        # NaNs arise so some regularization of log is required
        # last term is from normalization
        logit = torch.log(x + 1e-8) - torch.log(1 - x + 1e-8) + torch.log(1 - self.alpha) - torch.log(3)
        log_det = F.softplus(logit) + F.softplus(-logit)

        return logit, log_det.sum(dim=(1, 2, 3))

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            losses.append(self._step(batch).cpu().numpy())

        self.train()
        return np.mean(losses)

    def _step(self, batch):
        batch = batch.to(self.device).float()
        x, log_det = self.preprocess(batch)
        log_prob = self.log_prob(x)
        log_prob += log_det

        return -log_prob.mean() / (3 * 32 * 32)

    def _build_prior(self):
        self.bdist = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))

    def fit(self, trainloader, testloader, epochs=10, lr=1e-4):
        self._build_prior()
        losses = {"train": [], "test": [self._test(testloader)]}

        optim = Adam(self.parameters(), lr=lr)

        for _ in trange(epochs, desc="Training...", leave=False):
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses["train"].append(loss.detach().cpu().numpy())

            losses["test"].append(self._test(testloader))

        return losses

    @torch.no_grad()
    def sample(self, n):
        self.eval()
        z = self.bdist.sample((n, 3, 32, 32)).squeeze(-1)
        return self.sample2img(self.g(z))

    @torch.no_grad()
    def interpolate(self, images):
        """
        first half is start, second half is finish images, respectively.
        """
        self.eval()

        images = images.to(self.device).float()
        assert images.shape[0] % 2 == 0
        start_size = images.shape[0] // 2

        x, _ = self.preprocess(images)
        z, _ = self.f(x)

        latents = []
        for i in range(0, start_size):
            z_start = z[i].unsqueeze(0)
            z_finish = z[start_size + i].unsqueeze(0)

            d = (z_finish - z_start) / 5

            latents.append(z_start)
            for j in range(1, 5):
                latents.append(z_start + d * j)
            latents.append(z_finish)

        latents = torch.cat(latents)

        res = self.g(latents)
        return self.sample2img(res)

    @torch.no_grad()
    def sample2img(self, x):
        x = 1 / (1 + torch.exp(-x))
        x = x - self.alpha
        x = x / (1 - self.alpha)
        return x.cpu().numpy().transpose(0, 2, 3, 1)
