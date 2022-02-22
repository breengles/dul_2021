import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import trange


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def test(self, testloader):
        raise NotImplementedError

    def step(self, batch):
        raise NotImplementedError

    def fit(self, trainloader, testloader=None, epochs=1, lr=1e-4):
        optim = Adam(self.parameters(), lr=lr)

        losses = {"train": []}

        if testloader is not None:
            losses["test"] = [self.test(testloader)]

        for _ in trange(epochs, desc="Fitting..."):
            for batch in trainloader:
                loss = self.step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses["train"].append(loss.detach().cpu().numpy())

            if testloader is not None:
                losses["test"].append(self.test(testloader))

        for k, v in losses.items():
            losses[k] = np.array(v)

        return losses
