import torch
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm.notebook import tqdm, trange
import numpy as np
import torch.nn.functional as F


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = (
            torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, block_size=2, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()

        self.main = nn.Sequential(DepthToSpace(block_size), nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding))

    def forward(self, x):
        return self.main(torch.cat([x, x, x, x], dim=1))


class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, block_size=2, kernel_size=(3, 3), stride=1, padding=1, bias=False):
        super().__init__()

        self.s2d = SpaceToDepth(block_size)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x_ = sum(self.s2d(x).chunk(4, dim=1)) / 4.0
        return self.conv(x_)


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.residual = Upsample_Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1)
        self.shortcut = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x_ = self.main(x)
        return self.residual(x_) + self.shortcut(x)


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
        )
        self.residual = Downsample_Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1)
        self.shortcut = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x_ = self.main(x)
        return self.residual(x_) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, n_filters=256, z_dim=128):
        super().__init__()
        self.n_filters = n_filters
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 4 * 4 * n_filters)
        self.main = nn.Sequential(
            ResnetBlockUp(n_filters, n_filters),
            ResnetBlockUp(n_filters, n_filters),
            ResnetBlockUp(n_filters, n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
        )

        self.bdist = Normal(torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z):
        out = self.fc(z).reshape(-1, self.n_filters, 4, 4)
        return self.main(out)

    def sample(self, n_samples=100):
        z = self.bdist.sample([n_samples, self.z_dim]).to(self.device)
        return self(z)


class Critic(nn.Module):
    def __init__(self, n_filters=256):
        super().__init__()
        self.main = nn.Sequential(
            ResnetBlockDown(3, n_filters),
            ResnetBlockDown(n_filters, n_filters),
            ResnetBlockDown(n_filters, n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4, 4), padding=0),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.main(x).squeeze())


class SNGAN(nn.Module):
    def __init__(self, n_filters=128, lr=2e-4, n_critic=5, lam=10.0):
        super().__init__()
        self.n_critic = n_critic
        self.lam = lam

        self.generator = Generator(n_filters)
        self.critic = Critic(n_filters)

        self.g_optim = Adam(self.generator.parameters(), lr=lr, betas=(0, 0.9))
        self.c_optim = Adam(self.critic.parameters(), lr=lr, betas=(0, 0.9))

    @property
    def device(self):
        return next(self.parameters()).device

    def __critic_loss(self, real, fake):
        score_real = self.critic(real)
        score_fake = self.critic(fake)
        return -score_real.mean() + score_fake.mean() + self.lam * self.__gradient_penalty(real, fake)

    def __gradient_penalty(self, real, fake):
        """
        see algo 1 from https://arxiv.org/pdf/1704.00028.pdf
        """
        bs = real.shape[0]
        eps = torch.rand(bs, 1, 1, 1).to(self.device).expand_as(real)
        interps = eps * real + (1 - eps) * fake

        scores = self.critic(interps)

        grads = torch.autograd.grad(scores, interps, torch.ones_like(scores, device=self.device), create_graph=True)[
            0
        ].reshape(bs, -1)

        return torch.mean((torch.norm(grads, dim=1) - 1) ** 2)

    def fit(self, trainloader, n_iter):
        losses = []
        total_iters = 0
        epochs = n_iter // len(trainloader)
        # epochs = self.n_critic * n_iter // len(trainloader)

        g_scheduler = LambdaLR(self.g_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)
        c_scheduler = LambdaLR(self.c_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)

        for epoch in trange(epochs, desc="Training...", leave=False):
            for batch_real in tqdm(trainloader, desc="Batch", leave=False):
                total_iters += 1

                batch_real = batch_real.to(self.device)
                batch_fake = self.generator.sample(batch_real.shape[0])

                critic_loss = self.__critic_loss(batch_real, batch_fake)

                self.c_optim.zero_grad()
                critic_loss.backward()
                self.c_optim.step()

                losses.append(critic_loss.detach().cpu().numpy())

                if total_iters % self.n_critic == 0:
                    g_loss = -self.critic(self.generator.sample(batch_real.shape[0])).mean()

                    self.g_optim.zero_grad()
                    g_loss.backward()
                    self.g_optim.step()

            g_scheduler.step()
            c_scheduler.step()

        return np.array(losses)

    @torch.no_grad()
    def sample(self, n):
        return self.generator.sample(n)


class G(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, x_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z).reshape(-1, 1, 28, 28)


class D(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(x_dim + z_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, z):
        return self.main(torch.cat((x, z), dim=1))


class E(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        return self.main(x.reshape(x.shape[0], -1))


class BiGAN(nn.Module):
    def __init__(self, z_dim, x_dim=784, lr=2e-4, l2=2.5e-5):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.lr = lr

        self.generator = G(x_dim, z_dim)
        self.discriminator = D(x_dim, z_dim)
        self.encoder = E(x_dim, z_dim)
        self.cls = nn.Linear(z_dim, 10)

        self.g_optim = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2)
        self.d_optim = Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2)
        self.e_optim = Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2)
        self.cls_optim = Adam(self.cls.parameters(), lr=lr)

    def reset_cls(self):
        self.cls = nn.Linear(self.z_dim, 10)
        self.cls_optim = Adam(self.cls.parameters(), lr=self.lr)

    @property
    def device(self):
        return next(self.parameters()).device

    def adversarial_loss(self, x_real):
        bs = x_real.shape[0]

        z_fake = torch.randn(bs, self.z_dim).type_as(x_real)
        z_real = self.encoder(x_real).reshape(bs, -1)

        x_fake = self.generator(z_fake).reshape(bs, -1)
        x_real = x_real.reshape(bs, -1)

        return (
            -(self.discriminator(x_real, z_real)).log().mean() - (1 - self.discriminator(x_fake, z_fake)).log().mean()
        )

    def fit(self, trainloader, epochs):
        losses = []

        g_scheduler = LambdaLR(self.g_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)
        d_scheduler = LambdaLR(self.d_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)

        for epoch in trange(epochs, desc="Training", leave=False):
            batch_losses = []
            for batch_real, _ in tqdm(trainloader, desc="Batch...", leave=False):
                batch_real = batch_real.to(self.device)

                d_loss = self.adversarial_loss(batch_real)

                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                g_loss = -self.adversarial_loss(batch_real)

                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                losses.append(d_loss.detach().cpu().numpy())

            g_scheduler.step()
            d_scheduler.step()

        return np.array(losses)

    def fit_cls(self, trainloader, epochs):
        losses = []

        self.encoder.eval()

        for epoch in trange(epochs, desc="Train classifier...", leave=False):
            batch_losses = []
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    z = self.encoder(x)

                y_pred = self.cls(z)
                loss = F.cross_entropy(y_pred, y)

                self.cls_optim.zero_grad()
                loss.backward()
                self.cls_optim.step()

                batch_losses.append(loss.detach().cpu().numpy())

            losses.append(np.mean(batch_losses))

        self.train()

        return losses

    @torch.no_grad()
    def sample(self, n):
        self.generator.eval()
        z = (torch.rand(n, self.z_dim).to(self.device) - 0.5) * 2
        self.generator.train()
        return self.generator(z).reshape(-1, 1, 28, 28).cpu().numpy()

    @torch.no_grad()
    def recon(self, x):
        self.generator.eval()
        self.encoder.eval()

        z = self.encoder(x.to(self.device))
        recons = self.generator(z).reshape(-1, 1, 28, 28)

        self.train()

        return recons.cpu().numpy()
