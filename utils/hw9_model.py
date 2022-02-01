import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import trange


class Buffer:
    def __init__(self, max_size=10000, device="cpu"):
        self.max_size = max_size
        self.device = device

        self.images = torch.FloatTensor(max_size, 1, 28, 28).uniform_(-1.0, 1.0).to(device)

        self.filled_i = 0
        self.curr_size = 0

    def __len__(self):
        return self.curr_size

    def push(self, images):
        n = images.shape[0]
        if self.curr_size < self.max_size:
            self.curr_size += n

        if self.filled_i + n <= self.max_size:
            self.images[self.filled_i : self.filled_i + n] = images.to(self.device)
        else:
            free_i = self.max_size - self.filled_i
            self.images[self.filled_i :] = images[:free_i].to(self.device)
            self.images[: n - free_i] = images[free_i:].to(self.device)

        self.curr_size = min(self.max_size, self.curr_size + n)
        self.filled_i = (self.filled_i + n) % self.max_size

    def sample(self, batch_size, limit=False):
        if limit:
            indices = np.random.choice(self.curr_size, batch_size, replace=False)
        else:
            indices = np.random.choice(self.max_size, batch_size, replace=False)

        return self.images[indices]


class EBM(nn.Module):
    def __init__(self, alpha=0.1, buffer_size=8192, buffer_device="cpu"):
        super().__init__()
        self.alpha = alpha

        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 4),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.buffer = Buffer(buffer_size, buffer_device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.main(x)

    def set_model_grads(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def langevin_sample(self, batch_size, K=60, eps_init=10.0, noise=0.005):
        self.set_model_grads(False)
        self.eval()

        buffer_part_size = int(batch_size * 0.95)
        x0_buffer = self.buffer.sample(buffer_part_size).to(self.device)
        x0_noise = torch.FloatTensor(batch_size - buffer_part_size, 1, 28, 28).uniform_(-1.0, 1.0).to(self.device)

        x = torch.vstack((x0_buffer, x0_noise)).to(self.device)
        x.requires_grad = True

        for i in range(K):
            eps = eps_init - eps_init * i / K
            z = (torch.randn_like(x) * noise).to(self.device)

            grad_x = torch.autograd.grad(self(x).sum(), x)[0].clip(-0.03, 0.03)

            x = torch.clip(x + np.sqrt(2 * eps) * z + eps * grad_x, -1.0, 1.0)

        self.buffer.push(x.detach())

        self.set_model_grads(True)
        self.train()

        return x

    def _step(self, batch, noise=0.005):
        x_real = batch + torch.randn_like(batch) * noise
        x_fake = self.langevin_sample(batch.shape[0], noise=noise)

        loss_real = self(x_real.to(self.device))
        loss_fake = self(x_fake.to(self.device))

        contrastive_loss = loss_fake.mean() - loss_real.mean()
        reg_loss = self.alpha * (loss_real ** 2 + loss_fake ** 2).mean()

        return contrastive_loss, reg_loss

    def fit(self, trainloader, epochs=20, lr=1e-3, beta1=0.0, beta2=0.999):
        optim = Adam(self.parameters(), lr=lr, betas=(beta1, beta2))
        losses = {"contrastive": [], "regularization": []}

        for _ in trange(epochs, desc="Training...", leave=False):
            for batch in trainloader:
                images = batch[0]
                contrastive_loss, reg_loss = self._step(images)
                loss = contrastive_loss + reg_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses["contrastive"].append(contrastive_loss.detach().cpu().numpy())
                losses["regularization"].append(reg_loss.detach().cpu().numpy())

        return losses
