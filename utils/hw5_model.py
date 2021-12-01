import torch
from torch import nn
from tqdm.auto import tqdm, trange
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 2 * latent_dim),
        )

    def forward(self, X):
        mu, log_sigma = self.main(X).chunk(2, dim=1)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, X):
        out = self.fc(X)
        out = out.reshape(-1, 128, 4, 4)
        return self.conv(out)


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
        kld = -log_sigma - 0.5 + (torch.exp(2 * log_sigma) + mu ** 2) * 0.5
        return self.decoder(z), kld

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
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            recon_loss, kld_loss = self._step(batch)
            loss = recon_loss + kld_loss
            losses.append([loss.cpu().numpy(), recon_loss.cpu().numpy(), kld_loss.cpu().numpy()])

        return np.mean(losses, axis=0)

    def _step(self, batch):
        batch = batch.to(self.device)
        batch_recon, kld = self(batch)

        kld_loss = kld.sum(1).mean()
        rec_loss = F.mse_loss(batch_recon, batch, reduction="none").sum(1).mean()

        rec_loss + kld_loss

        return rec_loss, kld_loss

    def fit(self, trainloader, testloader, epochs, lr=1e-4, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        for _ in trange(epochs, desc="Training...", leave=True):
            train_losses = []

            losses["test"].append(self._test(testloader))

            for batch in trainloader:
                rec_loss, kld_loss = self._step(batch)

                loss = rec_loss + kld_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(
                    [loss.detach().cpu().numpy(), rec_loss.detach().cpu().numpy(), kld_loss.detach().cpu().numpy()]
                )

            losses["train"].append(np.mean(train_losses, axis=0))
            losses["test"].append(self._test(testloader))

        return np.array(losses["train"]), np.array(losses["test"])

    @torch.no_grad()
    def _convert2img(self, batch):
        out = (torch.clip(batch, -1, 1) * 0.5 + 0.5) * 255
        return out.cpu().numpy().transpose(0, 2, 3, 1)

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim).to(self.device)
        samples = torch.clamp(self.decode(z), -1, 1)
        return self._convert2img(samples)

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
