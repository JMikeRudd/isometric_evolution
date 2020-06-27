import torch
from torch.distributions import (
    MultivariateNormal, Categorical, Distribution)
from gym import spaces

from .embedding_models import (
    MixedEmbMapping, MLPEmbMapping, IDEmbMapping)

class Distn(torch.nn.Module):

    def __init__(self, target_dim, base_dist):
        super().__init__()
        assert isinstance(target_dim, int) and target_dim > 0
        self.target_dim = target_dim

        assert issubclass(base_dist, Distribution)
        self.base_dist = base_dist

    def _get_dists(self, **kwargs):
        params = self._validate_params(self._get_params(kwargs))
        return [self.base_dist(**p) for p in params]

    def _get_params(self, **kwargs):
        return NotImplementedError

    def _validate_params(self, params):
        return NotImplementedError

    def sample(self, **kwargs):
        dists = self._get_dists(**kwargs)
        samples = [d.sample([1]) for d in dists]
        return torch.cat([s.unsqueeze(0) for s in samples], dim=0)

    def log_prob(self, target, **kwargs):
        dists = self._get_dists(**kwargs)
        assert issubclass(type(target), torch.Tensor) and target.size(0) == len(dists)
        lps = [d.log_prob(target[i]) for (d, t) in zip(dists,list(target))]
        return torch.cat([lp.unsqueeze(0) for lp in lps], dim=0)


class CondDistn(Distn):
    '''
    Conditional distribution class, useful for policies
    '''
    def __init__(self, target_dim, base_dist, param_model):
        super().__init__(target_dim, base_dist)
        assert issubclass(type(param_model), MixedEmbMapping)
        self.param_model = param_model
        self.keys = param_model.keys

    def _get_params(self, cond_inputs, **kwargs):
        cond_inputs = self._validate_inputs(cond_inputs)
        return self.param_model(cond_inputs)

    def _validate_inputs(self, cond_inputs):
        return self.param_model._validate_inp(cond_inputs)


class DiscreteCondDistn(CondDistn):
    '''
    Conditional distribution over discrete variable
    '''
    def __init__(self, target_dim, param_model):
        super().__init__(target_dim, Categorical, param_model)
        assert self.param_model.emb_dim == self.target_dim
        self.sftmx = torch.nn.Softmax(dim=1)

    def _validate_params(self, params):
        assert issubclass(type(params), torch.Tensor)
        assert params.dim() == 2 and params.size(1) == self.target_dim
        norm_params = self.sftmx(params)
        return [{'probs': np} for np in list(norm_params)]


class GaussianCondDistn(CondDistn):
    '''
    Conditional distribution over discrete variable
    '''
    def __init__(self, target_dim, param_model):
        super().__init__(target_dim, MultivariateNormal, param_model)
        assert self.param_model.emb_dim == 2 * self.target_dim

    def _validate_params(self, params):
        assert isinstance(params, torch.FloatTensor)
        assert params.dim() == 2 and params.size(1) == self.target_dim
        mu, log_sigma = params.chunk(2, dim=1)
        sigma2 = (2 * log_sigma).exp()

        return [{'loc': m,
                 'covariance_matrix': torch.diag(s2)}
                 for (m, s2) in zip(list(mu), list(sigma2))]


def get_policy_distn(mdp, state_emb_model, goal_dim=None, distn_cls=None):

    # Get which type of policy to return
    if issubclass(type(mdp.action_space), spaces.Discrete):
        pol_cls = DiscreteCondDistn
        act_dim = mdp.action_space.n
        param_dim = act_dim
    elif issubclass(type(mdp.action_space), spaces.Box):
        pol_cls = GuassianCondDistn
        assert len(mdp.action_space.shape) == 1
        act_dim = mdp.action_space.shape[0]
        param_dim = 2 * act_dim
    else:
        raise ValueError('{} action space type not implemented'.format(
            type(mdp.action_space)))

    if goal_dim is None:
        # return a regular policy
        emb_dim = state_emb_model.emb_dim
        param_model = MixedEmbMapping(
            emb_model_dict={'obs': state_emb_model},
            emb_dim=param_dim,
            comb_model=MLPEmbMapping([emb_dim, 4*emb_dim, param_dim]))
    else:
        # return a goal-directed policy
        assert (isinstance(goal_dim, int) and
                goal_dim > 0 and
                goal_dim == state_emb_model.emb_dim)
        param_model = MixedEmbMapping(
            emb_model_dict={'obs': state_emb_model,
                            'goal': IDEmbMapping(goal_dim)},
            emb_dim=param_dim,
            comb_model=MLPEmbMapping([2*goal_dim, 4*goal_dim, param_dim]))
        return pol_cls(act_dim, param_model)

