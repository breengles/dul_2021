import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm.auto import tqdm, trange


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=128):
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
            nn.Linear(4 * 4 * 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(x.shape[0], -1)
        return self.linear(out)


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 1 x 28 x 28 -> 32 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # -> 64 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # -> 128 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),  # -> 128 x 4 x 4
            nn.ReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 128, latent_dim)

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(x.shape[0], -1)
        return self.linear(out)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
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
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(-1, 128, 4, 4)
        return self.convt(out)


class ContextEncoder(nn.Module):
    def __init__(self, latent_dim=128, crop_size=(14, 14)):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator(latent_dim)

        self.crop_size = crop_size
        self.eps = 1e-8

    @property
    def device(self):
        return next(self.parameters()).device

    def mask(self, img):
        crop_h, crop_w = self.crop_size
        batch_size, _, h, w = img.shape

        # x = [(h - crop_h) // 2]
        # y = [(w - crop_w) // 2]

        x = np.random.randint(0, h - crop_h, size=batch_size)
        y = np.random.randint(0, w - crop_w, size=batch_size)

        mask = torch.zeros_like(img)
        for img_mask, h, w in zip(mask, x, y):
            # [c, h, w]
            img_mask[:, h : h + crop_h, w : w + crop_w] = 1.0

        return mask.to(self.device)

    def recon_loss(self, x, recon, mask):
        recon_loss = (mask * F.mse_loss(recon, x, reduction="none")).reshape(x.shape[0], -1).mean(dim=1)

        # * min_F [log(D(x)) + log(1 - D(F))] ==> min_F log (1 - D(F))
        # f_loss = torch.log(1 - self.discriminator(recon) + self.eps)

        # * min log(1 - D(F)) ==> max log D(F) <==> min -log D(F)
        # f_loss = -torch.log(self.discriminator(recon))

        # * ???
        with torch.no_grad():
            fake = self.discriminator(recon)

        f_loss = F.binary_cross_entropy(fake, torch.ones_like(fake, device=self.device))

        ed_loss = recon_loss + f_loss

        return recon_loss.mean(), ed_loss.mean()

    def adversarial_loss(self, batch, recon):
        """
        L_adv = log(D(x)) + log(1 - D(F))
        """

        # loss_real = torch.log(self.discriminator(batch) + self.eps)
        # loss_fake = torch.log(1 - self.discriminator(recon) + self.eps)

        real = self.discriminator(batch)
        fake = self.discriminator(recon)

        loss_real = F.binary_cross_entropy(real, torch.ones_like(real, device=self.device))
        loss_fake = F.binary_cross_entropy(fake, torch.zeros_like(fake, device=self.device))

        loss = loss_real + loss_fake
        return loss.mean()

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def _step(self, batch):
        mask = self.mask(batch)
        recon = self.reconstruct((1 - mask) * batch)

        recon_loss, ed_loss = self.recon_loss(batch, recon, mask)
        d_loss = self.adversarial_loss(batch, recon.detach())

        return recon_loss, ed_loss, d_loss

    def fit(self, trainloader, epochs=10, lr=1e-3, discriminator_update_rate=10):
        encoder_optim = Adam(self.encoder.parameters(), lr=lr)
        decoder_optim = Adam(self.decoder.parameters(), lr=lr)
        discriminator_optim = Adam(self.discriminator.parameters(), lr=lr)

        recon_losses = []
        adversarial_losses = []

        iteration = 0
        for _ in trange(epochs, desc="Fitting..."):
            for batch in trainloader:
                iteration += 1
                batch = batch[0].float().to(self.device)

                recon_loss, ed_loss, d_loss = self._step(batch)

                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                ed_loss.backward()
                encoder_optim.step()
                decoder_optim.step()

                if (iteration + 1) % discriminator_update_rate == 0:
                    discriminator_optim.zero_grad()
                    d_loss.backward()
                    discriminator_optim.step()

                recon_losses.append(recon_loss.detach().cpu().numpy())
                adversarial_losses.append(d_loss.detach().cpu().numpy())

        return np.array(recon_losses), np.array(adversarial_losses)

    @torch.no_grad()
    def examples(self, x):
        self.eval()

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        mask = self.mask(x)
        x_curr = (1 - mask) * x
        x_rec = self.reconstruct(x_curr)

        out = np.vstack((x_curr.cpu().numpy(), x_rec.cpu().numpy(), x.cpu().numpy()))
        return out


class RNet(nn.Module):
    def __init__(self, hidden_dim=128) -> None:
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
            nn.Linear(4 * 4 * 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.rotations = [0, 90, 180, 270]
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.conv(x)
        return self.linear(out.reshape(x.shape[0], -1))

    @property
    def device(self):
        return next(self.parameters()).device

    def _step(self, batch):
        x, label = batch
        x = x.to(self.device)
        label = label.to(self.device)
        return self.criterion(self(x), label)

    @torch.no_grad()
    def _get_accuracy(self, loader):
        accuracies = []
        for batch in tqdm(loader, desc="Testing...", leave=False):
            x, label = batch
            x = x.to(self.device)

            y = torch.argmax(self(x), dim=1).cpu().numpy()

            accuracies.append(accuracy_score(label.cpu().numpy(), y))

        return np.mean(accuracies)

    def fit(self, trainloader, epochs=10, lr=1e-4):
        losses = []
        accuracy = [self._get_accuracy(trainloader)]

        optim = Adam(self.parameters(), lr=lr)

        for _ in trange(epochs, desc="Fitting..."):
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())

            accuracy.append(self._get_accuracy(trainloader))

        return np.array(losses), np.array(accuracy)

