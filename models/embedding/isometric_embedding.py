import os
import logging
from itertools import combinations
from tqdm import tqdm

from torch.nn import Module, MSELoss, Parameter
import torch

from .embedding_space import EmbeddingSpace
from .metrics import Metric


class IsometricEmbedding(Module):

    def __init__(self, emb_space, metric, proportional=False):

        super().__init__()

        assert issubclass(type(emb_space), EmbeddingSpace)
        self.emb_space = emb_space

        assert issubclass(type(metric), Metric)
        self.metric = metric

        self.loss_fn = MSELoss()

        # Only train the scale parameter if proportional is True
        assert isinstance(proportional, bool)
        self.K = Parameter(torch.ones(1), requires_grad=proportional)

        # Don't want to train any metric parameters
        # self.model_params = chain(self.K, self.emb_space.parameters())

    def forward(self, input):
        return None

    def real_distance(self, x, y):
        return self.metric(x, y)

    def latent_distance(self, x, y):
        return self.emb_space.compute_dist(x, y)

    def _train_batch(self, batch, optim, inp_tensor=None):
        optim.zero_grad()

        loss = 0
        if not isinstance(batch, dict):
            bs = batch.size(0)
        else:
            bs = [v for (k, v) in batch.items()][0].size(0)

        # Get unique pairs
        [I, J] = [list(idx) for idx in zip(*list(combinations(range(bs), 2)))]

        if not isinstance(batch, dict):
            x, y = batch[I], batch[J]
        else:
            x = {k: v[I] for (k, v) in batch.items()}
            y = {k: v[J] for (k, v) in batch.items()}

        # Get estimated distances in latent space and true distances
        if inp_tensor is None:
            est = self.latent_distance(x, y)
        else:
            assert issubclass(type(inp_tensor), torch.Tensor)
            est = self.latent_distance(inp_tensor[x], inp_tensor[y])

        real = self.real_distance(x, y)

        loss = self.loss_fn(self.K * est, real)
        loss.backward()
        optim.step()

        return loss.item()

    def embed(self, x):
        return self.emb_space.embed(x)


def train_isometric_embedding_epoch(isom_embedding, data_loader, optim, inp_tensor=None):
    epoch_losses = 0
    N = len(data_loader)

    for (_, batch) in enumerate(data_loader):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        if not isinstance(batch, dict):
            assert issubclass(type(batch), torch.Tensor)
        else:
            for (_, v) in batch.items():
                assert issubclass(type(v), torch.Tensor)

        batch_loss = isom_embedding._train_batch(batch, optim, inp_tensor=inp_tensor)

        epoch_losses += batch_loss / N

    return epoch_losses


def train_isometric_embedding(isom_embedding, epochs, data_loader, optim,
                              inp_tensor=None, print_every=1, save_every=1, save_dir=None):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for e in tqdm(range(epochs)):
        epoch_loss = train_isometric_embedding_epoch(
            isom_embedding, data_loader, optim, inp_tensor=inp_tensor)

        if e % print_every == 0:
            _log_losses(e, epoch_loss, logger)

        if e % save_every == 0:
            # save the model
            torch.save(isom_embedding,
                       os.path.join(save_dir, 'isometric_embedding'))


def _log_losses(epoch, epoch_loss, logger):

    log_msg = 'Epoch {}: {}'.format(epoch, epoch_loss)
    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)
