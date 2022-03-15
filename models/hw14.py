import torch
from torch import nn, pairwise_distance
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
from tqdm.auto import trange, tqdm


def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets


class ProtoNet(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()

        self.z_dim = z_dim

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, z_dim),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.main(x)

    @staticmethod
    def __pairwise_distance(x, y):
        return x.pow(2).sum(dim=1, keepdim=True) + y.pow(2).sum(dim=1) - 2 * x @ y.T

    def __get_proto(self, imgs, labels):
        bs = imgs.shape[0]

        uniq_labels = torch.unique(labels, sorted=False)  # turn off sorting to keep it as in batch
        n_uniq_labels = uniq_labels.shape[0]

        embeddings = self(imgs)

        protos = torch.zeros(n_uniq_labels, self.z_dim, device=self.device)

        for i, c in enumerate(uniq_labels):
            protos[i] = torch.mean(embeddings[labels == c], dim=0)

        return protos, uniq_labels

    def __loss(self, imgs, labels):
        imgs_s, imgs_q, labels_s, labels_q = split_batch(imgs, labels)

        protos, uniq_labels_s = self.__get_proto(imgs_s, labels_s)

        _, uniq_labels_q_idx = torch.unique(labels_q, return_inverse=True)

        emb = self(imgs_q)

        dist = self.__pairwise_distance(emb, protos)

        log_p = F.log_softmax(-dist, dim=-1)
        log_p = log_p[torch.arange(log_p.shape[0]), uniq_labels_q_idx]

        J = 0
        for k in uniq_labels_s:
            idx = labels_q == k
            #                          N_q         N_c
            J = J + log_p[idx].sum() / idx.sum() / uniq_labels_s.shape[0]

        return -J

    def __step(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        return self.__loss(imgs, labels)

    def fit(self, trainloader, epochs=1, lr=1e-4):
        optim = Adam(self.parameters(), lr=lr)

        losses = []

        for _ in trange(epochs):
            for batch in trainloader:
                loss = self.__step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())

        return np.array(losses)

    @torch.no_grad()
    def predict(self, batch, protos):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        pred_idxs = torch.argmin(self.__pairwise_distance(self(imgs), protos), dim=-1)

        return pred_idxs

    @torch.no_grad()
    def adapt_few_shots(self, batch, dataloader):
        self.eval()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        protos, uniq_labels = self.__get_proto(imgs, labels)

        pred = []

        for b in dataloader:
            pred_idxs = self.predict(b, protos)

            pred.append(uniq_labels[pred_idxs].cpu().numpy())

        return np.concatenate(pred)
