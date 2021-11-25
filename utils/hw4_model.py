import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from tqdm.auto import trange, tqdm
from torchdiffeq import odeint_adjoint as odeint


class UWBNet(nn.Module):
    def __init__(self, io_dim, hidden_dim=64, width=1):
        super().__init__()

        self.width = width
        self.io_dim = io_dim
        self.bs = io_dim * width

        self.main = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.bs * 2 + width),  # U W (2 * width * io) and B (width)
        )

    def forward(self, t):
        uwb = self.main(t.unsqueeze(0)).squeeze(0)

        u = uwb[: self.bs].reshape(self.width, 1, self.io_dim)
        w = uwb[self.bs : 2 * self.bs].reshape(self.width, self.io_dim, 1)
        b = uwb[-self.width :].reshape(self.width, 1, 1)

        return u, w, b


class CNF(nn.Module):
    def __init__(self, io_dim, hidden_dim=64, width=1):
        super().__init__()
        self.width = width
        self.params = UWBNet(io_dim, hidden_dim, width)

    @property
    def device(self):
        return self.params.main[0].weight.device

    @property
    def bdist(self):
        return MultivariateNormal(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))

    def forward(self, t, z):
        z = z[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.dz_dt(t, z)

            dlogpz_dt = self.dlogpz_dt(dz_dt, z)

        return dz_dt, dlogpz_dt

    def dz_dt(self, t, z):
        """
        dz_dt = u Tanh(w^T @ z + b)
        """
        u, w, b = self.params(t)
        z_ = z.unsqueeze(0).repeat(self.width, 1, 1)

        h = torch.tanh(z_.matmul(w) + b)
        return torch.mean(h.matmul(u), dim=0)

    def dlogpz_dt(self, f, z):
        return -self._trace(f, z)

    def _trace(self, f, z):
        trace = torch.zeros(z.shape[0], device=self.device)

        for i in range(z.shape[1]):
            trace += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]

        return trace.reshape(-1, 1)

    def flow(self, x, t0=0, t1=10):
        """
        log p_0(z_0) = log p_1(z_1) - \int_1^0 Tr(df / dz) dt
        """
        logdet_1 = torch.zeros((x.shape[0], 1)).to(self.device)

        z_t, logdet_t = odeint(
            self, (x, logdet_1), torch.FloatTensor([t1, t0]).to(self.device), atol=1e-5, rtol=1e-5, method="dopri5"
        )

        return z_t[-1], logdet_t[-1]

    def fit(self, trainloader, testloader, epochs, lr=1e-4, l2=0, t0=0, t1=10):
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses = {"train": [], "test": []}

        losses["test"].append(self._test(testloader, t0, t1))

        for _ in trange(epochs, desc="Fitting...", leave=True):
            train_losses = []
            for batch in trainloader:
                loss = self._step(batch, t0, t1)

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(self._test(testloader, t0, t1))

        return self, losses

    def _step(self, batch, t0=0, t1=10):
        batch = batch.to(self.device)
        return -self.log_prob(batch, t0, t1).mean()

    @torch.no_grad()
    def _test(self, testloader, t0=0, t1=10):
        losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            losses.append(self._step(batch, t0, t1).cpu().numpy())

        return np.mean(losses)

    def log_prob(self, batch, t0=0, t1=10):
        batch = batch.to(self.device)
        z, log_det = self.flow(batch, t0, t1)

        return self.bdist.log_prob(z) - log_det.flatten()

    @torch.no_grad()
    def densities(self, data, t0=0, t1=10):
        probas = []

        for batch in tqdm(data, desc="Getting densities...", leave=False):
            probas.append(self.log_prob(batch, t0, t1).exp().cpu().numpy())

        return np.hstack(probas)

    @torch.no_grad()
    def latent(self, data, t0=0, t1=10):
        latents = []

        for batch in tqdm(data, desc="Getting latents...", leave=False):
            batch = batch.to(self.device)
            latents.append(self.flow(batch, t0, t1)[0].cpu().numpy())

        return np.vstack(latents)


if __name__ == "__main__":
    t = torch.FloatTensor([1.0])
    net = UWBNet(2)

    net(t)
