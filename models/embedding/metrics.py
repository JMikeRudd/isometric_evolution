import torch
from copy import copy
from itertools import product
from torch.nn.functional import kl_div

from .utils import pi
from .embedding_models import MixedEmbMapping

METRICS = ['euclidean', 'angular', 'value_fn', '2dcurvesquaredist', 'ftJSD', 'min_dist']


class Metric(torch.nn.Module):
    ''' Class to hold all metrics, includes utilities for computing distances
        validating inputs, and updating the metric (if it is estimated)
        Arguments:
            None
        Methods:
            forward:
                Validate two inputs and compute distance between them. Inputs
                assumed to have shape (b, d) where b is batch size and d is the
                dimension of the inputs. d can change batch-by-batch as long as
                inputs have same dim and dim is not declared at initialization.
            _validate_inputs:
                Checks that inputs are torch tensors with the right dimension
            _compute_dist:
                Implemented by subclasses. Does the distance computation.
                Inputs same as forward. Returns tensor of scalar distances for
                each row in the batch.
            update_metric:
                Some metrics are estimated from data and will change over time.
                This method handles training on one batch and returns a dict of
                any info (e.g. loss) that the caller of the method might want.
    '''
    def __init__(self, dim=None):
        super().__init__()

        if dim is not None:
            assert isinstance(dim, int) and dim > 0
        self.dim = dim

    def forward(self, x, y):
        x, y = self._validate_inputs(x, y)
        return self._compute_dist(x, y)

    def _validate_inputs(self, x, y):
        assert (issubclass(type(x), torch.Tensor) and
                issubclass(type(y), torch.Tensor))
        if x.dim() == 1 and y.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        assert x.dim() == 2 and y.dim() == 2
        assert x.size(1) == y.size(1)
        if self. dim is not None:
            assert x.size(1) == self.dim
        return x, y

    def _compute_dist(self, x, y):
        raise NotImplementedError('Implemented by subclasses')

    def update_metric(self, batch):
        raise NotImplementedError('Implemented by subclasses')


class EuclideanMetric(Metric):
    ''' Metric is the Euclidean distance.
    '''
    def __init__(self, dim=None):
        super().__init__(dim)

    def _compute_dist(self, x, y):
        return ((x - y) ** 2).sum(dim=-1)

    def update_metric(self, batch):
        return {}


class AngularDistanceMetric(Metric):
    ''' Metric is the angular distance.
    '''
    def __init__(self, epsilon=0.001, dim=None):
        super().__init__(dim)

        assert isinstance(epsilon, float) and epsilon > 0
        self.epsilon = epsilon

    def _compute_dist(self, x, y):
        cos_sim = (x * y).sum(dim=-1) / (1 + self.epsilon)
        # assert cos_sim.max() < 1 and cos_sim.min() > -1

        return torch.acos(cos_sim) / pi

    def update_metric(self, batch):
        return {}


class ManhattanMetric(Metric):
    ''' Metric is the Manhattan distance.
    '''
    def __init__(self, dim=None):
        super().__init__(dim)

    def _compute_dist(self, x, y):
        return (x - y).abs().sum(dim=-1)

    def update_metric(self, batch):
        return {}


class AgentMetric(Metric):
    ''' Metric defined on agents.
    '''
    def __init__(self, unique_obs, dist_metric):
        super().__init__()

        assert issubclass(type(unique_obs), torch.Tensor)
        self.unique_obs = unique_obs
        self.n_states = unique_obs.size(0)

        self.dist_metric = dist_metric

    def _compute_dist(self, x, y):

        with torch.no_grad():
            x_pols = x.get_probs(self.unique_obs)
            y_pols = y.get_probs(self.unique_obs)

        return self.dist_from_pols(x_pols, y_pols)

    def dist_from_pols(self, x_pols, y_pols):
        return self.dist_metric(x_pols, y_pols).mean()

    def update_metric(self, batch):
        return {}


class JSDAgentMetric(AgentMetric):
    ''' Metric defined on agents.
    '''
    def __init__(self, unique_obs):
        super().__init__(unique_obs, JSDAgentMetric.JSD)

    @staticmethod
    def JSD(x_pol, y_pol):
        m_pol = 0.5 * (x_pol + y_pol)
        return 0.5 * (JSDAgentMetric._kl(x_pol, m_pol) + JSDAgentMetric._kl(y_pol, m_pol))

    @staticmethod
    def _kl(pol1, pol2):
        return (pol1 * (pol1.log() - pol2.log())).sum(dim=-1)

    def update_metric(self, batch):
        return {}


class TableMetric(Metric):
    ''' Metric is a lookup table of distances
    '''
    def __init__(self, table, dim=None):
        super().__init__(dim)
        assert issubclass(type(table), torch.Tensor)
        assert (table >= 0).all()
        assert table.dim() == 2 and table.size(0) == table.size(1)

        self.table = table

        if self.dim is None:
            self.dim = table.size(0)

    def _compute_dist(self, x, y):
        return self.table[x, y]

    def _validate_inputs(self, x, y):
        assert (issubclass(type(x), torch.Tensor) and
                issubclass(type(y), torch.Tensor))
        assert all([v.dtype in [torch.int32, torch.int64] for v in [x, y]])
        assert x.dim() == 1 and y.dim() == 1
        assert x.size(0) == y.size(0)
        assert (x >= 0).all() and (y >= 0).all()
        if self. dim is not None:
            assert (x < self.dim).all() and (y < self.dim).all()
        return x, y

    def update_metric(self, batch):
        return {}


class DictMetric(Metric):
    ''' Any metric meant to be applied to a dictionary of inputs.
        Methods:
            _validate_dict:
                Take dictionary as input, check it has appropriate keys,
                return only the needed info from it (assumed to be tensors)
    '''
    def __init__(self, dim=None):
        super().__init__(dim)

    def _validate_inputs(self, x, y):

        # Expect to receive dictionaries
        assert isinstance(x, dict) and isinstance(y, dict)

        x, y = self._validate_dict(x), self._validate_dict(y)
        return super()._validate_inputs(x, y)

    def _validate_dict(self, inp_dict):
        return NotImplementedError('Implemented by subclasses')

    def _compute_dist(self, x, y):
        raise NotImplementedError('Implemented by subclasses')

    def update_metric(self, batch):
        return {}


class TwoDCurveSquareDistMetric(DictMetric):
    ''' Metric is the integral of the square distance between two curves
        in some specified interval. Expect to receive polynomial coefficients
        of y in x.
    '''
    def __init__(self, interval=[-1., 1.]):
        super().__init__()

        assert isinstance(interval, list) and len(interval) == 2
        assert all([isinstance(b, float) for b in interval])
        self.lb = interval[0]
        self.ub = interval[1]

    def _validate_dict(self, inp_dict):
        assert 'coeffs' in inp_dict.keys()
        return inp_dict['coeffs']

    def _compute_dist(self, x, y):

        bs = x.size(0)
        max_deg = x.size(1)

        degs = torch.arange(max_deg)
        # square_degs = torch.zeros(max_deg, max_deg)
        int_vals = torch.zeros((bs, max_deg, max_deg))
        coeff_diff = (y - x)
        coeff_prods = torch.zeros((bs, max_deg, max_deg))

        if x.device.type == 'cuda' or y.device.type == 'cuda':
            degs = degs.cuda()
            int_vals = int_vals.cuda()
            coeff_diff = coeff_diff.cuda()
            coeff_prods = coeff_prods.cuda()

        for (i, j) in product(degs, degs):
            int_vals[:, i, j] = (self.ub ** (i + j + 1) -
                                 self.lb ** (i + j + 1)) / (i + j + 1).float()
            coeff_prods[:, i, j] = coeff_diff[:, i] * coeff_diff[:, j]

        return (int_vals * coeff_prods).sum(1).sum(1)


class MinDistMetric(Metric):
    '''
    Estimate distance between two states as the minimum expected time
    to travel between them.
    '''
    def __init__(self, value_fn, goal_model):
        super().__init__()
        assert issubclass(type(value_fn), GoalValueFunction)
        self.value_fn = value_fn

        assert issubclass(type(goal_model), MixedEmbMapping)
        self.goal_model = goal_model

    def _compute_dist(self, o1, o2):
        with torch.no_grad():
            g1 = self.goal_model({'obs': o1})
            g2 = self.goal_model({'obs': o2})
            d12 = -self.value_fn(obs=o1, goal=g2)
            d21 = -self.value_fn(obs=o2, goal=g1)
        return 0.5 * (d12 + d21)

    def update_metric(self, batch):
        return {}


class EstimatedMetric(Metric):
    ''' Subclass of all metrics that require updating/estimation.
        Assumes that the component to be estimated is a torch module
        updated by the data and that the distance is a metric computed
        on the outputs of the model.
        Arguments:
            model: torch.nn.Module
        Methods:
            _compute_loss (implemented by subclasses):
                Takes batch as input and computes loss to backprop to
                model parameters. Returns loss and and info (dict) the
                caller might want.
    '''
    def __init__(self, model, optim, model_metric):
        super().__init__()

        assert issubclass(type(model), torch.nn.Module)
        self.model = model

        assert issubclass(type(optim),
                          torch.optim.optimizer.Optimizer)
        self.optim = optim

        assert issubclass(type(model_metric), Metric)
        self.model_metric = model_metric

    def _compute_dist(self, x, y):
        mx, my = self.model(x), self.model(y)
        return self.model_metric(mx, my)

    def update_metric(self, batch):

        self.optim.zero_grad()

        # Compute loss
        batch_loss, info = self._compute_loss(batch)

        batch_loss.backward()
        self.optim.step()

        ret_dict = {'loss': batch_loss.item()}
        for k in info.keys():
            ret_dict[k] = info[k]

        return ret_dict

    def _compute_loss(self, batch):
        raise NotImplementedError('Implemented by subclasses')


class ValueFunctionMetric(EstimatedMetric):
    ''' Metric estimates the value of two states and then takes Euclidean
        distance of values.
    '''
    def __init__(self, model, optim, gamma=0.99):
        super().__init__(model, optim, EuclideanMetric(dim=1))

        assert isinstance(gamma, float) and gamma > 0 and gamma < 1
        self.gamma = gamma

        self.loss_fn = torch.nn.MSELoss()

    def _compute_loss(self, batch):

        # Unpack batch
        state = batch['state']
        next_state = batch['next_state']
        reward = batch['reward']
        done = batch['done']

        # Compute value estimate of current states
        value = self.model(state)

        # Compute Bellman estimate of value
        with torch.no_grad():
            next_value = self.model(next_state)

        target = reward + self.gamma * (1 - done) * next_value

        # Compute loss
        batch_loss = self.loss_fn(value, target)

        return batch_loss, {}


class ConditionalDistMetric(EstimatedMetric):
    ''' Metric estimates conditional distribution given states and
        computes some divergence given the parameters of those dists.
    '''
    def __init__(self, model, optim, divergence_metric_cls):
        super().__init__(
            model, optim,
            ConditionalDistMetric._init_divergence_metric(
                model, divergence_metric_cls))

        # self.loss_fn = torch.nn.MSELoss()
    @staticmethod
    def _init_divergence_metric(model, divergence_metric):
        ''' Divergence metric class must be a subclass of metric. Takes
            two sets of parameters and
        '''


    def _compute_loss(self, batch):

        # Unpack batch
        state = batch['state']
        next_state = batch['next_state']
        reward = batch['reward']
        done = batch['done']

        # Compute value estimate of current states
        value = self.model(state)

        # Compute Bellman estimate of value
        with torch.no_grad():
            next_value = self.model(next_state)

        target = reward + self.gamma * (1 - done) * next_value

        # Compute loss
        batch_loss = self.loss_fn(value, target)

        return batch_loss, {}


class DivergenceMetric(Metric):
    '''
    '''
    def __init__(self, dim=None):
        super().__init__(dim)


class LinearCombinationMetric(Metric):
    ''' Class for when the metric is a linear combination of other metrics.
        Restriction that each coefficient must be positive.
        Arguments:
            metrics (required):
                list of Metric objects
            weights (optional):
                list of floats. If None then defaults to 1/|metrics|
        Methods:
            forward:
                return linear combination of submetrics evaluated on inputs
            update_metric:
                return list of results of update_metric for submetrics
    '''
    def __init__(self, metrics, weights=None):
        super().__init()

        # Check that metrics passed as list
        assert isinstance(metrics, list)

        # If no weights given then assign each equal weight
        weights = [float(1 / len(metrics)) for _ in range(len(metrics))] if\
            weights is None else weights

        # Check weights are list equal in length to metrics
        assert isinstance(weights, list) and (len(metrics) == len(weights))

        # Check each metric is Metric subclass and w are all positive floats
        for (m, w) in zip(metrics, weights):
            assert issubclass(type(m), Metric)
            assert isinstance(w, float) and w > 0

        self.metrics = metrics
        self.weights = weights

        # Check that any metrics with dim not None have same dim
        dims = [m.dim for m in self.metrics if m.dim is not None]
        dim = dims[0] if len(dims) > 0 else None
        assert all([d == dim for d in dims])
        self.dim = dim

    def forward(self, x, y):

        dist = 0
        for (m, w) in zip(self.metrics, self.weights):
            dist += w * m(copy(x), copy(y))

        return dist

    def update_metric(self, batch):
        return [m.update_metric(batch) for m in self.metrics]


def get_metric(metric, model=None, optim=None, **metric_kwargs):
    assert metric in METRICS

    if metric == 'euclidean':
        return EuclideanMetric()
    elif metric == 'angular':
        return AngularDistanceMetric()
    elif metric == 'value_fn':
        assert model is not None and optim is not None
        gamma = metric_kwargs.get('gamma')
        return ValueFunctionMetric(model, optim, gamma=gamma)
    elif metric == '2dcurvesquaredist':
        return TwoDCurveSquareDistMetric()
