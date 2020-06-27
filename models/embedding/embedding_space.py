import torch
import logging

from .metrics import Metric, EuclideanMetric, AngularDistanceMetric, ManhattanMetric
from .embedding_models import EmbMapping

#logger = logging.getLogger(__name__)

EMBEDDING_SPACES = ['euclidean', 'spherical', 'manhattan']


class EmbeddingSpace(torch.nn.Module):
    ''' Base class embedding module. Basic properties of any embedding module are:
            1. Maps from input space X to latent space L
            2. Has a metric on L
            3. Can compute distances between inputs
        Arguments:
            mapping (embedding_models.EmbMapping, required): model that maps
                from X to L
        Methods:
            embed: given input x return mapped embedding l
            compute_dist: given two inputs x and y compute the distance
                d(lx,ly) in embedding space
            validate_emb: check that embedding is torch tensor, that it has
                shape (*, emb_dim), and that it has the correct topology
            check_topology (implemented by sublasses): check that an embedding
                is valid under the specified topology
            metric (implemented by sublasses): compute distance in embedding
                space
    '''
    def __init__(self, mapping, metric):

        super().__init__()

        assert issubclass(type(mapping), EmbMapping)
        self.mapping = mapping
        self.emb_dim = self.mapping.emb_dim
        self.inp_dim = self.mapping.inp_dim

        assert issubclass(type(metric), Metric)
        self.metric = metric

    def embed(self, x):
        return self.validate_emb(self.mapping(x))

    def compute_dist(self, x, y):
        lx, ly = self.embed(x), self.embed(y)

        assert lx.size(0) == ly.size(0)

        return self.metric(lx, ly)

    def validate_emb(self, lx):
        assert issubclass(type(lx), torch.Tensor)
        if lx.dim() == 1 and lx.size(0) == self.emb_dim:
            lx = lx.unsqueeze(0)
        assert lx.dim() == 2
        assert lx.size(1) == self.emb_dim
        return self.check_topology(lx)

    def check_topology(self, lx):
        return NotImplementedError('Implemented by subclasses')


class EuclideanEmbeddingSpace(EmbeddingSpace):
    ''' Embedding model with Riemannian topology and Euclidean metric
    '''
    def __init__(self, mapping):
        super().__init__(mapping, EuclideanEmbeddingSpace._get_metric(mapping))
        self.type = 'euclidean'

    @staticmethod
    def _get_metric(mapping):
        assert issubclass(type(mapping), EmbMapping)
        return EuclideanMetric(dim=mapping.emb_dim)

    def check_topology(self, lx):
        return lx


class SphericalEmbeddingSpace(EmbeddingSpace):
    ''' Embedding model with spherical topology and angular distance metric
    '''
    def __init__(self, mapping):
        super().__init__(mapping, SphericalEmbeddingSpace._get_metric(mapping))
        self.type = 'spherical'

    @staticmethod
    def _get_metric(mapping):
        assert issubclass(type(mapping), EmbMapping)
        return AngularDistanceMetric(dim=mapping.emb_dim)

    def check_topology(self, lx):
        return lx / lx.norm(dim=-1).unsqueeze(-1)


class ManhattanEmbeddingSpace(EmbeddingSpace):
    ''' Embedding model with Riemannian topology and Manhattan distance
    '''
    def __init__(self, mapping):
        super().__init__(mapping, ManhattanEmbeddingSpace._get_metric(mapping))
        self.type = 'manhattan'

    @staticmethod
    def _get_metric(mapping):
        assert issubclass(type(mapping), EmbMapping)
        return ManhattanMetric(dim=mapping.emb_dim)

    def check_topology(self, lx):
        return lx


def get_embedding_space(emb_space_type, mapping):
    assert emb_space_type in EMBEDDING_SPACES

    if emb_space_type == 'euclidean':
        return EuclideanEmbeddingSpace(mapping)
    elif emb_space_type == 'spherical':
        return SphericalEmbeddingSpace(mapping)
    elif emb_space_type == 'manhattan':
        return ManhattanEmbeddingSpace(mapping)
