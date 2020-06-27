import torch
import numpy as np
from copy import deepcopy

from ..embedding.embedding_models import EmbMapping
from ..agent import GoalValueFunction

from .envs import ENVS, get_mdp


# Functions assume batches
def goal_directed_reward(obs, act, next_obs, goal_obs, batch=True, tol=0.001):
    if batch:
        abs_diff = (obs - goal_obs).abs()
        for _ in range(obs.dim() - 1):
            abs_diff = abs_diff.sum(dim=-1)
        return (1 * (abs_diff <= tol).float()) - 1
    else:
        if (obs - goal_obs).abs().sum() <= tol:
            return 0
        else:
            return -1

def goal_directed_termination(obs, act, next_obs, goal_obs, batch=True, tol=0.001):

    if batch:
        curr_abs_diff = (obs - goal_obs).abs()
        next_abs_diff = (next_obs - goal_obs).abs()
        for _ in range(obs.dim() - 1):
            curr_abs_diff = curr_abs_diff.sum(dim=-1)
            next_abs_diff = next_abs_diff.sum(dim=-1)
        #return (curr_abs_diff <= tol or next_abs_diff <= tol)
        return (curr_abs_diff <= tol) + (next_abs_diff <= tol)
    else:
        #return (obs - goal_obs).abs().sum() <= tol or (next_obs - goal_obs).abs().sum() <= tol
        return (obs - goal_obs).abs().sum() <= tol or (next_obs - goal_obs).abs().sum() <= tol


class GoalSampler(object):
    '''
    Base class for all methods to sample goal space
    '''
    def __init__(self, goal_dim):
        assert isinstance(goal_dim, int) and goal_dim > 0
        self.goal_dim = goal_dim

    def sample_goal(self, batch, **kwargs):
        raise NotImplementedError('Implemented by subclasses')


class InitDistGoalSampler(GoalSampler):
    '''
    Sample state from initial distribution and embed
    '''
    def __init__(self, mdp, emb_model, presamples=0, device='cpu'):
        super().__init__(InitDistGoalSampler._init_sampler(mdp, emb_model))

        self.mdp = mdp
        self.emb_model = emb_model

        assert device in ['cpu', 'cuda']
        self.device = device

        # Moving new goals to gpu slows computation and since this distribution
        # doesn't change over training we can sample ahead of time (memory cost)
        assert isinstance(presamples, int) and presamples >= 0
        if presamples > 0:
            self.sample_lib = torch.cat(
                [self.mdp.reset().unsqueeze(0) for _ in range(presamples)],
                dim=0).to(self.device)
        else:
            self.sample_lib = None

    @staticmethod
    def _init_sampler(mdp, emb_model):
        assert issubclass(type(emb_model), EmbMapping)
        goal_dim = emb_model.emb_dim
        return goal_dim

    def to(self, device):
        assert device in ['cpu', 'cuda']
        self.device = device
        if self.sample_lib is not None:
            self.sample_lib = self.sample_lib.to(device)

    def cuda(self):
        self.to('cuda')

    def sample_goal(self, batch):
        n_goals = batch['obs'].size(0)
        
        if self.sample_lib is None:
            goal_obs = [self.mdp.reset().unsqueeze(0) for _ in range(n_goals)]
            goal_obs = torch.cat(goal_obs, dim=0).to(self.device)
        else:
            samp_idxs = list(np.random.choice(self.sample_lib.size(0), n_goals, replace=True))
            goal_obs = self.sample_lib[samp_idxs]

        with torch.no_grad():
            goals = self.emb_model(goal_obs)
        return goal_obs, goals


class NextObsGoalSampler(GoalSampler):
    '''
    Goal is next obs. Used for curriculum learning so rewards less sparse
    '''
    def __init__(self, emb_model):
        super().__init__(NextObsGoalSampler._init_sampler(emb_model))
        self.emb_model = emb_model

    @staticmethod
    def _init_sampler(emb_model):
        assert issubclass(type(emb_model), EmbMapping)
        goal_dim = emb_model.emb_dim
        return goal_dim

    def sample_goal(self, batch):
        goal_obs = batch['next_obs']
        with torch.no_grad():
            goal = self.emb_model(goal_obs)
        return goal_obs, goal


class EmbDistGoalSampler(GoalSampler):
    '''
    Goal state cannot be more than specified (emb space) distance from start
    '''
    def __init__(self, mdp, emb_model, value_fn):
        super().__init__(EmbDistGoalSampler._init_sampler(mdp, emb_model))

        self.mdp = mdp
        self.emb_model = emb_model
        self.value_fn = value_fn

    @staticmethod
    def _init_sampler(mdp, emb_model, value_fn):
        assert issubclass(type(emb_model), EmbMapping)
        goal_dim = emb_model.emb_dim
        return goal_dim

    def sample_goal(self, batch):
        device = 'cuda' if next(self.emb_model.parameters()).is_cuda else 'cpu'
        n_goals = batch['obs'].size(0)
        goal_obs = [self.mdp.reset().unsqueeze(0).to(device) for _ in range(n_goals)]
        goal_obs = torch.cat(goal_obs, dim=0)
        with torch.no_grad():
            goals = self.emb_model(goal_obs)
        return goal_obs, goals


class MixtureGoalSampler(GoalSampler):
    '''
    Combines list of goal samplers
    '''
    def __init__(self, goal_samplers):
        super().__init__(MixtureGoalSampler._init_sampler(goal_samplers))

        self.goal_samplers = goal_samplers
        self.n_components = len(goal_samplers)

    @staticmethod
    def _init_sampler(goal_samplers):
        assert isinstance(goal_samplers, list) and len(goal_samplers) > 1
        assert all([issubclass(type(gs), GoalSampler) for gs in goal_samplers])

        goal_dim = goal_samplers[0].goal_dim
        assert all([gs.goal_dim == goal_dim for gs in goal_samplers])

        return goal_dim

    def sample_goal(self, batch, mixture_weights=None):

        if mixture_weights is None:
            mixture_weights = torch.ones(self.n_components).float()
            mixture_weights /= self.n_components

        if isinstance(mixture_weights, float):
            assert self.n_components == 2
            assert 0 <= mixture_weights <= 1
            mixture_weights = torch.tensor(
                [mixture_weights, (1 - mixture_weights)]).float()

        assert isinstance(mixture_weights, torch.Tensor)
        assert (mixture_weights >= 0).all() and (mixture_weights <= 1).all()
        
        mixture_weights = mixture_weights.numpy()
        mixture_weights /= np.sum(mixture_weights)

        component = np.random.choice(self.n_components, 1, p=mixture_weights)[0]
        return self.goal_samplers[component].sample_goal(batch)


class GoalMDPWrapper(object):
    """ docstring for GoalMDPWrapper
        Creates set of tasks from MDP where each task corresponds to moving to a
        specifc goal state.
    """
    def __init__(self, env_name, n_tasks, goal_states=None):
        super(GoalMDPWrapper, self).__init__()
        assert isinstance(env_name, str) and env_name in ENVS
        self.env_name = env_name
        self.env, _, _, _ = get_mdp(env_name)

        assert isinstance(n_tasks, int) and n_tasks > 0
        self.n_tasks = n_tasks

        if goal_states is None:
            self.goal_states = torch.cat([self.env.reset().unsqueeze(0) for _
                                          in range(self.n_tasks)], dim=0)
        else:
            assert issubclass(type(goal_states), torch.Tensor) and goal_states.size(0) == self.n_tasks
            self.goal_states = goal_states

    def save(self, save_path):
        torch.save({'env_name': self.env_name, 'n_tasks': self.n_tasks, 'goal_states': self.goal_states}, save_path)

    @classmethod
    def load(cls, load_path):
        kwargs = torch.load(load_path)
        return cls(**kwargs)

    def reset(self):
        # choose task
        self.curr_task_id = np.random.choice(self.n_tasks, 1)
        self.curr_goal = self.goal_states[self.curr_task_id]

        # reset base env
        self.obs = self.env.reset()

        return deepcopy(self.obs), torch.tensor(deepcopy(self.curr_task_id)).long()

    def step(self, action):

        # take step in base env
        next_obs, _, _, _ = self.env.step(action)

        # compute reward and termination for current goal
        reward = goal_directed_reward(self.obs, action, next_obs, self.curr_goal, batch=False)
        done = goal_directed_termination(self.obs, action, next_obs, self.curr_goal, batch=False)

        # update obs
        self.obs = next_obs
        return next_obs, reward, done, {'task_id': self.curr_task_id}


def get_goal_mdp_wrapper(env_name, n_tasks):
    _, obs_type, obs_dim, act_dim = get_mdp(env_name)
    return GoalMDPWrapper(env_name, n_tasks), obs_type, obs_dim, act_dim
