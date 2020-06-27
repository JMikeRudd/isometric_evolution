import gym
from .custom.gym_minigrid.wrappers import *
from .custom.simple_grid import SimpleGrid, SimpleGridGoal
import torch
import numpy as np

ENVS = ['fourrooms', 'simple_grid_4', 'simple_grid_10', 'minigrid_6', 'minigrid_16', 'simple_grid_goal']

def get_mdp(env_name):
    assert env_name in ENVS
    if env_name == 'fourrooms':
        env = gym.make('MiniGrid-FourRooms-v0', )
        env.seed(np.random.randint(0, 1000000))
        env = RGBImgObsWrapper(env) # Get pixel observations
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        env = TorchImageMDPWrapper(env, norm_const=255.)
        obs_type = 'cont'
        obs_dim = env.observation_space.shape
        act_dim = 3 # env.action_space.n
    elif env_name == 'simple_grid_4':
        env = SimpleGrid(size=4)
        obs_type = 'cont'
        obs_dim = env.dim
        act_dim = 4
    elif env_name == 'simple_grid_10':
        env = SimpleGrid()
        obs_type = 'cont'
        obs_dim = env.dim
        act_dim = 4
    elif env_name == 'minigrid_16':
        env = gym.make('MiniGrid-Empty-Random-16x16-v0', )
        env.seed(np.random.randint(0, 1000000))
        env = RGBImgObsWrapper(env) # Get pixel observations
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        env = TorchImageMDPWrapper(env, norm_const=255.)
        obs_type = 'cont'
        obs_dim = env.observation_space.shape
        act_dim = 3 # env.action_space.n
    elif env_name == 'minigrid_6':
        env = gym.make('MiniGrid-Empty-Random-6x6-v0', )
        env.seed(np.random.randint(0, 1000000))
        env = RGBImgObsWrapper(env) # Get pixel observations
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        env = TorchImageMDPWrapper(env, norm_const=255.)
        obs_type = 'cont'
        obs_dim = env.observation_space.shape
        act_dim = 3 # env.action_space.n
    elif env_name == 'simple_grid_goal':
        env = SimpleGridGoal()
        obs_type = 'cont'
        obs_dim = env.dim
        act_dim = 4
    else:
        raise ValueError('{} is not a recognized environment'.format(env_name))

    return env, obs_type, obs_dim, act_dim


class TorchMDPWrapper(object):
    '''
    Wrapper class to make mdp input and output compatible with torch models
    '''
    def __init__(self, mdp):
        self.mdp = mdp
        self.observation_space = mdp.observation_space
        self.action_space = mdp.action_space

    def reset(self):
        obs = self.mdp.reset()
        return self._process_obs(torch.tensor(obs).float())

    def step(self, act):
        obs, rew, done, info = self.mdp.step(act.cpu().numpy())
        return self._process_obs(torch.tensor(obs).float()), rew, done, info

    def _process_obs(self, obs):
        return obs


class TorchImageMDPWrapper(TorchMDPWrapper):
    '''
    Overwrite process _obs_method for image data b/c torch models expect
    channels first
    '''
    def __init__(self, mdp, norm_const=1.):
        super().__init__(mdp)
        assert isinstance(norm_const, float) or isinstance(norm_const, int) and norm_const > 0
        self.norm_const = float(norm_const)

        self.observation_space = gym.spaces.box.Box(low=np.transpose(mdp.observation_space.low, [2, 0, 1]),
                                                    high=np.transpose(mdp.observation_space.high, [2, 0, 1]) / norm_const)

    def _process_obs(self, obs):
        assert issubclass(type(obs), torch.Tensor)
        return obs.float().transpose(0,2) / self.norm_const
