import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from tqdm.auto import trange


class MaskedLinear(nn.Linear):
    """same as Linear except has a configurable mask on the weights"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, hidden_sizes: Union[List, Tuple], inp_dim, out_dim, d):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.d = d

        layers = []
        sizes = [inp_dim * d] + list(hidden_sizes) + [out_dim * d]

        for s1, s2 in zip(sizes[:-1], sizes[1:-1]):
            layers.extend([MaskedLinear(s1, s2), nn.ReLU()])
        else:
            layers.append(MaskedLinear(sizes[-2], sizes[-1]))

        self.main = nn.Sequential(*layers)

        self.m = {}
        self.make_masks()

    def make_masks(self):
        L = len(self.hidden_sizes)

        self.m[-1] = np.repeat(np.arange(self.inp_dim), self.d)
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(), self.inp_dim - 1, size=self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        masked_layers = [l for l in self.main.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(masked_layers, masks):
            l.set_mask(m)

    def forward(self, X):
        X_ = self.main(X).reshape(X.shape[0], self.out_dim, self.d)
        X_ = F.softmax(X_, dim=-1)
        return X_

    def __step(self, batch):
        criterion = nn.BCELoss()
        batch = batch.to(self.main[0].weight.device)

        # criterion = nn.NLLLoss()
        # outs = self(batch).reshape(batch.size(0), self.d, self.out_dim)
        # classes = torch.argmax(batch.reshape(batch.size(0), self.out_dim, self.d), dim=-1)
        # loss = criterion(outs, classes)

        outs = self(batch)
        loss = criterion(outs, batch.reshape(batch.size(0), self.out_dim, self.d))
        return loss

    def __test(self, test):
        test_losses = []
        with torch.no_grad():
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

    def sample(self):
        device = self.main[0].weight.device
        with torch.no_grad():
            x = torch.zeros((1, self.inp_dim, self.d)).to(device)
            for it in range(self.out_dim):
                out = self(x.reshape(1, -1)).squeeze(0)
                curr_hist = out[it].numpy()
                idx = np.random.choice(self.d, p=curr_hist)
                x[:, it, idx] = 1

        return x
