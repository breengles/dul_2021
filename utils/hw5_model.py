import torch
from torch import nn
from tqdm.auto import tqdm, trange
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),  # 2 x 2
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),  # 1 x 1
            nn.ReLU(),
            nn.Conv2d(128, 2 * latent_dim, 1, 1, 0),
        )

    def forward(self, x):
        mu, log_sigma = self.main(x).chunk(2, dim=1)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x):
        return self.main(x)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        z = mu + sigma * torch.randn_like(sigma)
        return self.decode(z), mu, log_sigma

    def encode(self, x):
        mu, log_sigma = self.encoder(x)
        z = mu + log_sigma.exp() * torch.randn_like(log_sigma)
        return z

    def decode(self, z):
        return self.decoder(z)

    @property
    def device(self):
        return self.encoder.main[0].weight.device

    @torch.no_grad()
    def _test(self, testloader):
        losses = []
        recon_losses = []
        kld_losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            loss, recon_loss, kld_loss = self._step(batch)

            losses.append(loss.cpu().numpy())
            recon_losses.append(recon_loss.cpu().numpy())
            kld_losses.append(kld_loss.cpu().numpy())

        return np.mean(losses), np.mean(recon_losses), np.mean(kld_losses)

    def _step(self, batch):
        batch = batch.to(self.device)

        batch_recon, mu, log_sigma = self(batch)

        rec_loss = F.mse_loss(batch_recon, batch, reduction="none").reshape(batch.shape[0], -1).sum(1).mean()
        kld_loss = (0.5 * (-2 * log_sigma - 1 + torch.exp(2 * log_sigma) + mu ** 2)).sum(1).mean()
        loss = rec_loss + kld_loss

        return loss, rec_loss, kld_loss

    def fit(self, trainloader, testloader, epochs, lr=1e-4, l2=0):
        losses = {"train": [], "test": []}
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Training...", leave=True):
            for batch in trainloader:
                loss, rec_loss, kld_loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # train losses every batch
                losses["train"].append(
                    [loss.detach().cpu().numpy(), rec_loss.detach().cpu().numpy(), kld_loss.detach().cpu().numpy()]
                )

            # test losses every epoch
            losses["test"].append(self._test(testloader))

        return np.array(losses["train"]), np.array(losses["test"])

    @torch.no_grad()
    def _convert2img(self, batch):
        out = (torch.clip(batch, -1, 1) + 1) * 255.0 / 2
        return out.cpu().numpy().transpose(0, 2, 3, 1)

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim, 1, 1).to(self.device)
        return self._convert2img(self.decode(z))

    @torch.no_grad()
    def reconstruct(self, batch):
        batch = batch.to(self.device)

        z = self.encode(batch)

        batch_recon = self.decode(z)

        reconstructions = torch.stack((batch, batch_recon), dim=1).reshape(-1, 3, 32, 32)

        return self._convert2img(reconstructions)

    @torch.no_grad()
    def interps(self, batch, n=10):
        batch = batch.to(self.device)

        z = self.encode(batch)

        zi, zf = z.chunk(2, dim=0)

        interps = [self.decode(zi * (1 - x) + zf * x) for x in np.linspace(0, 1, n)]
        interps = torch.stack(interps, dim=1).reshape(-1, 3, 32, 32)

        return self._convert2img(interps)
