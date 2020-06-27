import time
import logging

import torch
from torch.nn.functional import softmax
from torch.distributions import Uniform, Categorical

from .embedding.embedding_models import EmbMapping, MixedEmbMapping
from .embedding.embedding_space import EmbeddingSpace
from .embedding.distns import CondDistn

from .rl.utils import onehot, DEFAULT_GAMMA


class GoalValueFunction(torch.nn.Module):
    '''
    Maps states/goal-states to expected future rewards
    '''
    def __init__(self, emb_space):
        super().__init__()
        assert issubclass(type(emb_space), EmbeddingSpace)
        self.emb_space = emb_space

    def value(self, obs, goal_obs):
        return -1 * self.emb_space.compute_dist(obs, goal_obs)

    def value_loss(self, obs, goal_obs, next_obs, reward, done, **kwargs):
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            next_val_est = (1 - (1 * done.float())) * self.value(next_obs, goal_obs)
            val_est = reward + (1 - (1 * done.float())) * next_val_est 
        val = self.value(obs, goal_obs)
        return (0.5 * (val - val_est) ** 2).mean(), val.detach(), next_val_est.detach()


class Policy(torch.nn.Module):
    '''
    Select action given state
    Arguments:
        obs_emb_model (not trainable):
            Module that embeds the input observation
        policy_comb_net (trainable component):
            Module that takes embedded observation and processes it with
            any other inputs to get the output
    '''
    def __init__(self, policy_comb_net, obs_emb_model):
        super().__init__()
        assert issubclass(type(policy_comb_net), torch.nn.Module)
        self.policy_comb_net = policy_comb_net

        assert issubclass(type(obs_emb_model), EmbMapping)
        self.obs_emb_model = obs_emb_model
        self.keys = ['obs']

    def forward(self, obs, **kwargs):
        # Do not want policy to train the embedding model
        with torch.no_grad():
            emb_obs = self.obs_emb_model(obs)

        # process obs further (trainable component)
        return self._comb_obs(obs, emb_obs, **kwargs)

    def _comb_obs(self, emb_obs, **kwargs):
        return NotImplementedError('implemented by subclasses')

    def get_action(self, **kwargs):
        return NotImplementedError('implemented by subclasses')

    def policy_loss(self, gamma=1., **kwargs):
        return NotImplementedError('implemented by subclasses')


'''
class StochasticPolicy(Policy):
    #Policy is distribution over actions given state
    def __init__(self, policy_net):
        super().__init__(policy_net)
        assert issubclass(type(policy_net), CondDistn)
        self.keys = policy_net.keys

    def get_action(self, **kwargs):
        return self.policy_net.sample(**kwargs)

    def policy_loss(self, **kwargs):
        # Policy Gradient update?
        return NotImplementedError('implemented by subclasses')
'''

class QPolicy(Policy):
    '''
    Policy is Q function
    '''
    def __init__(self, policy_comb_net, obs_emb_model, value_fn):
        super().__init__(policy_comb_net, obs_emb_model)

        assert issubclass(type(policy_comb_net), MixedEmbMapping)
        assert 'emb_obs' in policy_comb_net.keys
        self.keys = policy_comb_net.keys

        assert issubclass(type(value_fn), GoalValueFunction)
        self.value_fn = value_fn

        self.unif = Uniform(0, 1)
        self.cat = Categorical(
            probs=torch.ones(policy_comb_net.emb_dim) / policy_comb_net.emb_dim)

    def _comb_obs(self, obs, emb_obs, **kwargs):

        # Prepare dict input expected by policy_comb_net
        policy_comb_net_inp = {'obs': obs, 'emb_obs': emb_obs}
        for k in self.keys:
            if k not in ['obs', 'emb_obs']:
                assert k in kwargs.keys()
                policy_comb_net_inp[k] = kwargs[k]

        return self.policy_comb_net(policy_comb_net_inp)

    def get_action(self, epsilon=0., **kwargs):
        assert isinstance(epsilon, float) and 0 <= epsilon <= 1
        if self.unif.sample([1]) > epsilon:
            # choose action with highest q-value
            self.policy_comb_net.eval()
            q_vals = self(**kwargs)
            act = q_vals.argmax(-1).cpu()
        else:
            # randomly select action
            act = self.cat.sample([1])
        return act

    def get_q_vals(self, obs, **kwargs):
        return self(obs, **kwargs)

    def value(self, obs, **kwargs):

        if 'goal' in kwargs.keys():
            assert 'gamma' in kwargs.keys()
            gamma = kwargs['gamma']
            # can use goal-directed value function
            with torch.no_grad():
                emb_obs = self.obs_emb_model(obs)
                dist = self.value_fn.emb_space.metric(emb_obs, kwargs['goal']).round().int()

            # Value is trained with gamma = 1 for interpretability, convert to equivalent value with current gamma
            if gamma != 1.:
                return -1 * (1 - gamma ** dist) / (1 - gamma)
            else:
                return -1 * dist
        else:
            # compute max q-value
            with torch.no_grad():
                q_vals = self(obs, **kwargs)
            return q_vals.max(dim=-1)[0]

    def policy_loss(self, gamma, obs, act, reward, done, next_obs, **kwargs):

        self.policy_comb_net.train()
        device = 'cuda' if next(self.policy_comb_net.parameters()).is_cuda else 'cpu'

        # get Q-Value estimates
        #q_start = time.time()
        q_vals = self(obs, **kwargs)
        #print('q', time.time() - q_start)

        # get q-value of action taken
        q = q_vals.gather(1, act)

        # Get best q-value from next observation
        #next_q_start = time.time()
        with torch.no_grad():
            next_val_est = self.value(next_obs, gamma=gamma, **kwargs)
            #next_q_vals = self(next_obs, **kwargs)
            #best_next_q_vals = next_q_vals.max(dim=-1)[0]
            q_est = reward + gamma * (1 - (1 * done.float())) * next_val_est
        #print('next_q', time.time() - next_q_start)
        
        return (0.5 * (q - q_est) ** 2).mean()


class StochPolicy(torch.nn.Module):
    '''
    Policy is a distribution over actions conditional on state.
    Assumes discrete actions
    '''
    def __init__(self, policy_model):
        super().__init__()

        assert issubclass(type(policy_model), MixedEmbMapping)
        assert 'obs' in policy_model.keys
        self.policy_model = policy_model
        self.keys = policy_model.keys
        self.act_dim = policy_model.emb_dim

        '''
        self.unif = Uniform(0, 1)
        self.cat = Categorical(
            probs=torch.ones(policy_comb_net.emb_dim) / policy_comb_net.emb_dim)
        '''

    def get_probs(self, obs, **kwargs):

        # Prepare dict input expected by policy_comb_net
        policy_model_inp = {'obs': obs}
        for k in self.keys:
            if k not in ['obs']:
                assert k in kwargs.keys()
                policy_model_inp[k] = kwargs[k]

        return softmax(self.policy_model(policy_model_inp), dim=-1)

    def get_action(self, **kwargs):
        pol_probs = self.get_probs(**kwargs)

        assert pol_probs.dim() == 2
        acts = torch.zeros(pol_probs.size(0)).long()
        for i in range(pol_probs.size(0)):
            acts[i] = Categorical(probs=pol_probs[i]).sample([1])

        return acts

class GoalDirectedStochPolicy(StochPolicy):
    '''
    Policy is a distribution over actions conditional on state and goal
    '''
    def __init__(self, policy_model):
        super().__init__(policy_model)

        assert 'emb' in policy_model.keys


'''
class GoalDirectedPolicy(Policy):
    # Maps state-goal pairs to distribution over actions
    def __init__(self, policy_distn):
        super().__init__(policy_distn)
        assert 'goal' in policy_distn.keys
'''

class Agent(torch.nn.Module):
    '''
    Wrapper class for policy including action sampling computation of value function
    Arguments:
        policy (required):
            Policy class instance with neural network for mapping states to
            actions
        value_fn (required):
            ValueFunction class instance with neural network mapping states to
            expected rewards. In this project all policies are expected to have
            a value function as it is used for distance estimation.
    '''
    def __init__(self, policy, value_fn):
        super().__init__()

        assert issubclass(type(policy), Policy)
        self.policy = policy

        assert issubclass(type(value_fn), GoalValueFunction)
        self.value_fn = value_fn

    def value(self, **kwargs):
        return self.value_fn.value(**kwargs)

    def get_action(self, **kwargs):
        return self.policy.get_action(**kwargs)

    def batch_loss(self, batch):
        raise NotImplementedError('implemented by subclasses')


class GoalDirectedAgent(Agent):
    '''
    Low level policy that takes a goal and current state and converts it into a policy
    over atomic actions
    '''
    def __init__(self, policy, value_fn):
        super().__init__(policy, value_fn)

        assert 'goal' in policy.keys
        self.policy = policy

        assert issubclass(type(value_fn), GoalValueFunction)
        self.value_fn = value_fn

    def batch_loss(self, batch):
        raise NotImplementedError('implemented by subclasses')


class GoalDirectedQAgent(GoalDirectedAgent):
    '''
    Goal-directed agent with q-function policy
    '''
    def __init__(self, qpolicy, value_fn):
        super().__init__(qpolicy, value_fn)
        assert issubclass(type(qpolicy), QPolicy)

    def batch_loss(self, batch, gamma=DEFAULT_GAMMA, train_pol=True, train_val=True):
        
        #logger = logging.getLogger(__name__)
        #logger.setLevel(logging.INFO)

        assert isinstance(batch, dict)
        #v_start = time.time()
        if train_val:
            v_loss, _, _ = self.value_fn.value_loss(**batch)
        else:
            v_loss = 0.
        #logger.info('value: {}'.format(time.time() - v_start))
        #p_start = time.time()
        if train_pol:
            p_loss = self.policy.policy_loss(gamma=gamma, **batch)
        else:
            p_loss = 0.
        #logger.info('policy: {}'.format(time.time() - p_start))

        return p_loss, v_loss
