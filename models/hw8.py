import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from tqdm.auto import tqdm, trange


class MomentMatchDRE(nn.Module):
    def __init__(self, hd=128, sigma=1):
        super().__init__()
        self.r = nn.Sequential(
            nn.Linear(1, hd), nn.ReLU(), nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 1), nn.Softplus(),
        )

        self.sigma = sigma

    @property
    def device(self):
        return next(self.parameters()).device

    def K(self, x, y):
        return (-(x.unsqueeze(-1) - y).squeeze() ** 2 / (2 * self.sigma)).exp()

    def fit(self, loader_nu, loader_de, lr=1e-3, num_epochs=1000):
        optim = Adam(self.r.parameters(), lr=lr)

        for _ in tqdm(range(num_epochs)):
            for batch_nu, batch_de in zip(loader_nu, loader_de):
                batch_nu = batch_nu.view(-1, 1).float().to(self.device)
                batch_de = batch_de.view(-1, 1).float().to(self.device)

                n_nu = batch_nu.shape[0]
                n_de = batch_de.shape[0]

                r_de = self.r(batch_de)

                K_dede = self.K(batch_de, batch_de)
                K_denu = self.K(batch_de, batch_nu)

                loss = 1 / n_de ** 2 * r_de.T @ K_dede @ r_de - 2 / (n_nu * n_de) * r_de.T @ K_denu @ torch.ones(
                    n_nu, device=self.device
                )

                optim.zero_grad()
                loss.backward()
                optim.step()

    @torch.no_grad()
    def predict(self, batch):
        return self.r(batch.to(self.device)).cpu().numpy()


class Classifier(nn.Module):
    def __init__(self, latent_dim=1, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * 4 * 128 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, z):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)

        return self.linear(torch.cat((out, z), dim=1))


class Encoder(nn.Module):
    def __init__(self, latent_dim=1, noise_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(4 * 4 * 128 + noise_dim, latent_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)

        return self.linear(torch.cat((out, noise), dim=1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 128), nn.ReLU())

        self.convt = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.convt(self.linear(x).reshape(x.shape[0], 128, 4, 4))


class AVB(nn.Module):
    def __init__(self, latent_dim=64, noise_dim=16):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim, noise_dim=noise_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.clf = Classifier(latent_dim=latent_dim)

        self.noise_dist = MultivariateNormal(torch.zeros(noise_dim), torch.eye(noise_dim))
        self.z_dist = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def _gen_loss(self, batch):
        noise = self.noise_dist.sample((batch.shape[0],)).to(self.device)
        z_encoded = self.encoder(batch, noise)

        T = self.clf(batch, z_encoded).mean()
        recon = self.decoder(z_encoded)
        loss = T + F.binary_cross_entropy_with_logits(recon, batch, reduction="sum") / batch.shape[0]
        return loss

    def _clf_loss(self, batch):
        z = self.z_dist.sample((batch.shape[0],)).to(self.device)
        noise = self.noise_dist.sample((batch.shape[0],)).to(self.device)
        z_encoded = self.encoder(batch, noise)

        T_real = self.clf(batch, z_encoded)
        T_fake = self.clf(batch, z)
        real_labels = torch.ones_like(T_real)
        fake_labels = torch.zeros_like(T_fake)
        loss_real = F.binary_cross_entropy_with_logits(T_real, real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(T_fake, fake_labels)
        return loss_real + loss_fake

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        elbo_losses = 0
        clf_losses = 0

        for batch in tqdm(testloader, desc="Testing...", leave=False):
            batch = batch.to(self.device)
            elbo_loss = self._gen_loss(batch)
            clf_loss = self._clf_loss(batch)
            elbo_losses += elbo_loss
            clf_losses += clf_loss

        self.train()

        return elbo_losses.cpu().numpy().mean(), clf_losses.cpu().numpy().mean()

    def fit(self, trainloader, testloader, epochs: int = 20, lr: float = 1e-5):
        train_losses = []
        test_losses = []

        decoder_optim = Adam(self.decoder.parameters(), lr=lr)
        encoder_optim = Adam(self.encoder.parameters(), lr=lr)
        clf_optim = Adam(self.clf.parameters(), lr=lr)

        test_losses.append(self._test(testloader))

        for _ in trange(epochs, desc="Training"):
            for batch in trainloader:
                batch = batch.to(self.device)

                elbo_loss = self._gen_loss(batch)

                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                elbo_loss.backward()
                encoder_optim.step()
                decoder_optim.step()

                clf_loss = self._clf_loss(batch)
                clf_optim.zero_grad()
                clf_loss.backward()
                clf_optim.step()

                train_losses.append((elbo_loss.detach().cpu().numpy(), clf_loss.detach().cpu().numpy()))

            test_losses.append(self._test(testloader))

        return np.array(train_losses), np.array(test_losses)

    @torch.no_grad()
    def _tensor2image(self, tensor):
        return tensor.clip(0, 1).cpu().numpy()

    @torch.no_grad()
    def sample(self, n):
        z = self.z_dist.sample((n,)).to(self.device)
        return self._tensor2image(self.decoder(z))
