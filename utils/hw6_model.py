import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from tqdm.auto import trange, tqdm
from torch.distributions import Normal


class Encoder(nn.Module):  # copied from hw5
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


class Decoder(nn.Module):  # copied from hw5
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


class MaskedConv(nn.Conv2d):  # copied from hw2
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


class MaskedResidualBlock(nn.Module):  # copied from hw2
    def __init__(self, inc, color=True, dc=3):
        super().__init__()
        h = inc // 2

        self.main = nn.Sequential(
            nn.ReLU(),
            MaskedConv(color=color, dc=dc, isB=True, in_channels=inc, out_channels=h, kernel_size=1),
            nn.ReLU(),
            MaskedConv(color=color, dc=dc, isB=True, in_channels=h, out_channels=h, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv(color=color, dc=dc, isB=True, in_channels=h, out_channels=inc, kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class MaskedLinear(nn.Linear):  # copied from hw1
    """same as Linear except has a configurable mask on the weights"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):  # copied from hw1
    def __init__(self, inp_dim, d, hidden_sizes: Union[List, Tuple] = (512, 512)):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.inp_dim = np.prod(inp_dim)
        self.out_dim = self.inp_dim * d
        self.d = d

        layers = []
        sizes = [inp_dim] + list(hidden_sizes) + [self.out_dim]
        for s1, s2 in zip(sizes[:-1], sizes[1:-1]):
            layers.extend([MaskedLinear(s1, s2), nn.ReLU()])
        else:
            layers.append(MaskedLinear(sizes[-2], sizes[-1]))

        self.main = nn.Sequential(*layers)

        self.m = {}
        self.make_masks()

        self.criterion = nn.CrossEntropyLoss()

    def make_masks(self):
        L = len(self.hidden_sizes)

        self.m[-1] = np.arange(self.inp_dim)
        self.m["out"] = np.repeat(np.arange(self.inp_dim), self.d)
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(), self.inp_dim - 1, size=self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m["out"][None, :])

        masked_layers = [l for l in self.main.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(masked_layers, masks):
            l.set_mask(m)

    def forward(self, X):
        return self.main(X.reshape(-1, self.inp_dim)).reshape(-1, self.inp_dim, self.d)  # this reshape works correctly

    def predict_proba(self, X):
        return F.softmax(self(X), dim=-1)

    def __step(self, batch):
        batch = batch.to(self.main[0].weight.device)
        outs = self(batch.reshape(batch.size(0), -1)).transpose(1, 2)  # this reshape works correctly
        classes = torch.argmax(batch, dim=-1)
        loss = self.criterion(outs, classes)

        return loss

    @torch.no_grad()
    def __test(self, test):
        test_losses = []

        for batch in test:
            loss = self.__step(batch)
            test_losses.append(loss.cpu().numpy())

        return np.mean(test_losses)

    def fit(self, train, test, epochs=100, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        # test before train
        losses["test"].append(self.__test(test))

        for _ in trange(epochs, desc="Fitting...", leave=False):
            train_losses = []

            # train
            for batch in train:
                loss = self.__step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.detach().cpu().numpy())

            # test
            test_losses = self.__test(test)

            losses["train"].append(np.mean(train_losses))
            losses["test"].append(test_losses)

        return self, np.array(losses["train"]), np.array(losses["test"])

    @torch.no_grad()
    def sample(self):
        device = self.main[0].weight.device

        x = torch.zeros((self.inp_dim, self.d)).to(device)

        for it in range(self.out_dim):
            out = self.predict_proba(x.flatten().unsqueeze(0)).squeeze(0)
            curr_hist = out[it].numpy()
            idx = np.random.choice(self.d, p=curr_hist)
            x[it, idx] = 1

        return x

    @property
    def device(self):
        return next(self.main.parameters()).device

    @property
    def bdist(self):
        return Normal(
            torch.tensor(0, dtype=torch.float32, device=self.device),
            torch.tensor(1, dtype=torch.float32, device=self.device),
        )

    def log_prob(self, z):
        mu, log_sigma = self(z).chunk(2, dim=-1)

        log_sigma = torch.tanh(log_sigma)

        mu, log_sigma = mu.squeeze(), log_sigma.squeeze()

        eps = z.squeeze().reshape(-1, self.inp_dim) * log_sigma.exp() + mu

        return self.bdist.log_prob(eps) + log_sigma


class PixelCNN(nn.Module):  # copied from hw2
    def __init__(self, input_shape, code_size, dim, cf=120):
        super().__init__()
        self.embedding = nn.Embedding(code_size, dim)
        self.input_shape = input_shape
        self.dc = 1
        self.code_size = code_size

        self.model = nn.Sequential(
            MaskedConv(color=False, dc=self.dc, isB=False, in_channels=dim, out_channels=cf, kernel_size=7, padding=3),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedResidualBlock(cf, False, dc=self.dc),
            MaskedConv(color=False, dc=self.dc, isB=True, in_channels=cf, out_channels=cf, kernel_size=1),
            nn.ReLU(),
            MaskedConv(color=False, dc=self.dc, isB=True, in_channels=cf, out_channels=code_size, kernel_size=1),
        )

    def forward(self, x):
        # return self.model(x).reshape(x.shape[0], self.num_classes, self.dc, *self.input_shape)

        # print(self.embedding(x).shape)

        out = self.embedding(x).permute(0, 3, 1, 2)
        return self.model(out).reshape(x.shape[0], self.dc, self.code_size, *self.input_shape).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def predict_proba(self, x):
        return F.softmax(self(x), dim=1)

    @property
    def device(self):
        return self.model[0].weight.device

    def _step(self, batch):
        batch = batch.to(self.device)
        return F.cross_entropy(self(batch), batch.unsqueeze(1).long())

    @torch.no_grad()
    def _test(self, testloader):
        losses = []

        for batch in tqdm(testloader, desc="Testing...", leave=False):
            losses.append(self._step(batch).cpu().numpy())

        return np.mean(losses)

    def fit(self, trainloader, testloader, epochs=20, lr=1e-3, l2=0):
        losses = {"train": [], "test": []}

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        # test before train
        losses["test"].append(self._test(testloader))

        for epoch in trange(epochs, desc="Fitting...", leave=True):
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

        return losses

    @torch.no_grad()
    def sample(self, n):
        sample = torch.zeros(n, *self.input_shape, dtype=torch.long).to(self.device)

        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                probs = self.predict_proba(sample)[..., i, j].squeeze()
                sample[:, i, j] = torch.multinomial(probs, 1).flatten()

        return sample
        # return sample.cpu().numpy().transpose(0, 1, 2)


class VLAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.MADE = MADE(latent_dim, 2)

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        z = mu + sigma * torch.randn_like(sigma)
        return self.decode(z), z, mu, log_sigma

    def encode(self, x):
        mu, log_sigma = self.encoder(x)
        z = mu + log_sigma.exp() * torch.randn_like(log_sigma)
        return z

    def decode(self, z):
        return self.decoder(z)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        losses = []
        recon_losses = []
        kld_losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            loss, recon_loss, kld_loss = self._step(batch)

            losses.append(loss.cpu().numpy())
            recon_losses.append(recon_loss.cpu().numpy())
            kld_losses.append(kld_loss.cpu().numpy())

        self.train()
        return np.mean(losses), np.mean(recon_losses), np.mean(kld_losses)

    def _kld_loss(self, z, mu, log_sigma):
        dist = Normal(mu, log_sigma.exp())
        log_q_z_x = dist.log_prob(z).squeeze().reshape(-1, self.latent_dim)  # encoder
        log_p_z = self.MADE.log_prob(z)  # prior

        return (log_q_z_x - log_p_z).sum(1).mean()

    def _step(self, batch):
        batch = batch.to(self.device)

        batch_recon, z, mu, log_sigma = self(batch)

        rec_loss = F.mse_loss(batch_recon, batch, reduction="none").reshape(batch.shape[0], -1).sum(1).mean()

        kld_loss = self._kld_loss(z, mu, log_sigma)

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

        for i in trange(self.latent_dim, desc="Sampling...", leave=False):
            mu, log_sigma = self.MADE(z)[:, i].chunk(2, dim=-1)

            mu, log_sigma = mu.unsqueeze(-1), log_sigma.unsqueeze(-1)
            log_sigma = torch.tanh(log_sigma)

            z[:, i] = (z[:, i] - mu) * (-log_sigma).exp()

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


class CodeBook(nn.Module):
    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / size, 1.0 / size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        n, _, h, w = z.shape

        with torch.no_grad():
            z_ = z.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
            distances = (
                (z_ ** 2).sum(dim=1, keepdim=True)
                - 2 * torch.matmul(z_, self.embedding.weight.t())
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
            )
            encoding_indices = torch.argmin(distances, dim=1).reshape(n, h, w)

        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2)

        return quantized, (quantized - z).detach() + z, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, code_dim, code_size, beta=1.0):
        super().__init__()

        self.code_dim = code_dim
        self.code_size = code_size
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, 2, 1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = CodeBook(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def code_encoding(self, batch):
        z = self.encoder(batch)
        return self.codebook(z)[-1]

    @torch.no_grad()
    def code_decoding(self, batch):
        z = self.codebook.embedding(batch).permute(0, 3, 1, 2)
        return self.decoder(z).permute(0, 2, 3, 1)

    def forward(self, x):
        z = self.encoder(x)
        e, e_, _ = self.codebook(z)
        x_decoded = self.decoder(e_)

        return z, e, x_decoded

    def _step(self, batch):
        batch = batch.to(self.device)

        z, e, batch_recon = self(batch)

        recon_loss = F.mse_loss(batch_recon, batch)
        vq_loss = F.mse_loss(e, z.detach())  # is it working?
        commitment_loss = F.mse_loss(z, e.detach())

        return recon_loss + vq_loss + self.beta * commitment_loss

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()

        test_losses = []
        for batch in tqdm(testloader, desc="Testing...", leave=False):
            loss = self._step(batch)
            test_losses.append(loss.cpu().numpy())

        self.train()

        return np.mean(test_losses)

    def fit(self, trainloader, testloader, epochs, lr=1e-4, l2=0):
        losses = {"train": [], "test": []}
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        losses["test"].append(self._test(testloader))

        for _ in trange(epochs, desc="Training...", leave=True):
            for batch in trainloader:
                loss = self._step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # train losses every batch
                losses["train"].append(loss.detach().cpu().numpy())

            # test losses every epoch
            losses["test"].append(self._test(testloader))

        losses["train"] = np.array(losses["train"])
        losses["test"] = np.array(losses["test"])

        return losses
