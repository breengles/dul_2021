import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet
from torch import autograd, nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange


@torch.no_grad()
def show_imgs(x, norm=True):
    x = x.detach().cpu()

    img = make_grid(x, nrow=5)
    img = img.permute(1, 2, 0)

    if norm:
        img = img * 0.5 + 0.5

    plt.imshow(img.numpy())
    plt.show()


def f_(f, x):
    x = x.clone()

    with torch.enable_grad():
        if not x.requires_grad:
            x.requires_grad = True

        y = f(x)

        (grad,) = autograd.grad(y.sum(), x, create_graph=False)

    return grad


@torch.no_grad()
def solve_sde(x, f, g, ts=0, tf=1, dt=1e-3, device="cuda"):
    for t in tqdm(np.arange(ts, tf, dt)):
        tt = torch.FloatTensor([t]).to(device)
        z = torch.randn_like(x).to(device)
        x = x + f(x, tt) * dt + g(tt) * z * abs(dt) ** 0.5

    return x


class ContDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps_th = Unet(dim=16, dim_mults=(1, 2, 3), channels=1)

    @property
    def device(self):
        return next(self.parameters()).device

    def gamma(self, t):
        return torch.log(torch.expm1(1e-4 + 10 * t ** 2))

    # Coefficient \bar{a}_t
    # see presentation, slide 19
    def a_bar(self, t):
        g = self.gamma(t)
        return torch.sigmoid(-g)

    def sigma_2(self, t):
        return 1 - self.a_bar(t)

    def log_a_bar_sqrt_(self, t):
        return 0.5 * f_(lambda s: torch.log(self.a_bar(s)), t)

    # Coefficient f(x, t)
    # see presentation, slide 31
    def f(self, x_t, t):
        return self.log_a_bar_sqrt_(t) * x_t

    # Coefficient g^2(t)
    # see presentation, slide 31
    def g_2(self, t):
        return f_(self.sigma_2, t) - 2 * self.log_a_bar_sqrt_(t) * self.sigma_2(t)

    # Learned score function
    # see presentation, slide 28
    def score(self, x_t, t):
        """
        x_t: tensort [bs, 1, 16, 16]
        t: tensort [bs]

        Returns
        - score: tensor  [bs, 1, 16, 16]
        """

        eps = -self.eps_th(x_t, t)
        sigma = self.sigma_2(t).sqrt().reshape(-1, 1, 1, 1)
        return eps / sigma

    def sample_t(self, bs):
        device = self.device
        t = torch.rand(bs).to(device)
        return t

    # Transition sampling q(x_t|x_0)
    # see presentation, slide 19 and 25
    def sample_x_t(self, x_0, t):
        """
        x_0: tensort [bs, 1, 16, 16]
        t: tensort [bs]

        Returns
        - x_t: tensor  [bs, 1, 16, 16]
        """
        alpha_bar = self.a_bar(t).reshape(-1, 1, 1, 1)

        eps = torch.randn_like(x_0, device=self.device)
        x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps

        return x_t, eps

    # Loss function
    # see presentation, slide 26
    def get_loss(self, x_0):
        bs = x_0.shape[0]
        data_dims = tuple(np.arange(1, len(x_0.shape)))

        t = self.sample_t(bs)
        x_t, eps = self.sample_x_t(x_0, t)

        loss = ((eps - self.eps_th(x_t, t)) ** 2).sum(dim=data_dims)
        loss = loss.mean()

        return loss

    # Sampling according to reverse SDE
    # see presentation, slide 32
    # Hint: use solve_sde function
    def sample_sde(self, bs):
        """
        bs: int

        Returns
        - x_0: tensor  [bs, 1, 16, 16] generated data
        """

        # starting from absolute noise
        x_t = torch.randn((bs, 1, 16, 16), device=self.device)

        # dx = [f(x, t) - g^2(t) * score(x, t)]dt + g(t) dw
        # solve_sde: fdt + gdw ==> f := [f(x, t) - g^2(t) * score(x, t)], g := g(t)
        f = lambda x_t, t: self.f(x_t, t) - self.g_2(t) * self.score(x_t, t)
        g = lambda t: self.g_2(t).sqrt()
        # g = lambda t: self.g_2(t) * self.score(x, eps, t)

        x_0 = solve_sde(x_t, f, g, ts=1, tf=0, dt=-1e-3, device=self.device)
        return x_0

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        losses = []
        for batch, _ in testloader:
            losses.append(self._step(batch).cpu().numpy())

        self.train()
        return np.mean(losses)

    def _step(self, x):
        x = x.to(self.device)
        loss = self.get_loss(x)
        return loss

    def fit(self, trainloader, testloader, epochs=15, lr=1e-4):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in (bar := trange(epochs)) :
            losses = []
            for batch, _ in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())

            bar.set_description(f"train: {np.mean(losses):0.4f} | val: {self._test(testloader):0.4f}")


class SinusoidalPosEmb(nn.Module):
    # from denoising_diffusion_pytorch
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Clf(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        # this one also from denoising_diffusion_pytorch
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

        self.x = nn.ModuleList(
            [nn.Linear(16 * 16 + dim, 8 * 8), nn.Linear(8 * 8 + dim, 4 * 4), nn.Linear(4 * 4 + dim, 10),]
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, t):
        if t.shape[0] == 1:
            t = t.repeat(x.shape[0])

        t_emb = self.time_mlp(t)

        x = x.flatten(1)
        for m in self.x[:-1]:
            x = torch.cat([x, t_emb], dim=1)
            x = F.relu(m(x))

        x = torch.cat([x, t_emb], dim=1)
        return self.x[-1](x)

    def gamma(self, t):
        return torch.log(torch.expm1(1e-4 + 10 * t ** 2))

    def a_bar(self, t):
        g = self.gamma(t)
        return torch.sigmoid(-g)

    def sample_t(self, bs):
        return torch.rand(bs).to(self.device)

    def sample_x_t(self, x_0, t):
        alpha_bar = self.a_bar(t).reshape(-1, 1, 1, 1)

        eps = torch.randn_like(x_0, device=self.device)
        x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps

        return x_t

    def _step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        t = self.sample_t(x.shape[0])
        x = self.sample_x_t(x, t)

        logit = self(x, t)
        loss = F.cross_entropy(logit, y)
        return loss

    @torch.no_grad()
    def _test(self, testloader):
        losses = []
        for batch in testloader:
            losses.append(self._step(batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=10, lr=1e-4):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        print(f"Train loss: {None} | test loss {self._test(testloader)}")

        for epoch in (bar := trange(epochs)) :
            train_losses = []
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            bar.set_description(f"Train loss: {np.mean(train_losses)} | test loss {self._test(testloader)}")


class CondContDDPM(ContDDPM):
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def score(self, x_t, t, target):
        part1 = super().score(x_t, t)

        with torch.enable_grad():
            x_t.requires_grad = True
            y = self.clf(x_t, t)[:, target]
            part2 = autograd.grad(y.sum(), x_t, create_graph=False)[0]

        x_t.requires_grad = False

        return part1 + part2

    # Sampling according to reverse SDE
    # see presentation, slide 32
    # Hint: use solve_sde function
    def sample_sde(self, bs, target):
        """
        bs: int

        Returns
        - x_0: tensor  [bs, 1, 16, 16] generated data
        """
        # starting from absolute noise
        x_t = torch.randn((bs, 1, 16, 16), device=self.device)

        # dx = [f(x, t) - g^2(t) * score(x, t)]dt + g(t) dw
        # solve_sde: fdt + gdw ==> f := [f(x, t) - g^2(t) * score(x, t)], g := g(t)
        f = lambda x_t, t: self.f(x_t, t) - self.g_2(t) * self.score(x_t, t, target)
        g = lambda t: self.g_2(t).sqrt()
        # g = lambda t: self.g_2(t) * self.score(x, eps, t)

        x_0 = solve_sde(x_t, f, g, ts=1, tf=0, dt=-1e-3, device=self.device)
        return x_0
