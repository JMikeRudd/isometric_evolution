import torch
import numpy as np
from copy import deepcopy

class SimpleGrid(object):

    def __init__(self, size=10, dim=2):
        assert isinstance(size, int) and size > 0
        self.size = size

        assert isinstance(dim, int) and dim > 0
        self.dim = dim
        assert dim == 2 # (for now don't bother with this)

    def reset(self, pos=None):
        if pos is None:
            # randomly init state within bounds
            self.obs = torch.tensor(np.random.choice(self.size, self.dim)).long()
        else:
            # custom place obs at start state
            assert isinstance(pos, tuple) and len(pos) == 2
            for j in range(self.dim):
                assert 0 <= pos[j] < self.size
            self.obs = torch.tensor(pos).long()

        return self._process_obs(self.obs)

    def step(self, act):
        if issubclass(type(act), torch.Tensor):
            assert act.dim() == 1 and len(act) == 1
            act = int(act[0])

        assert isinstance(act, int) and 0 <= act <= 3

        '''
        0 = left
        1 = up
        2 = right
        3 = down
        '''

        old_obs = deepcopy(self.obs)
        if act == 0:
            self.obs[0] = torch.max(self.obs[0] - 1, torch.tensor([0]))
        elif act == 1:
            self.obs[1] = torch.min(self.obs[1] + 1, torch.tensor([self.size - 1]))
        elif act == 2:
            self.obs[0] = torch.min(self.obs[0] + 1, torch.tensor([self.size - 1]))
        elif act == 3:
            self.obs[1] = torch.max(self.obs[1] - 1, torch.tensor([0]))

        return (self._process_obs(self.obs),
                self._process_reward(obs=old_obs, new_obs=self.obs, act=act),
                self._process_done(obs=old_obs, new_obs=self.obs, act=act),
                {})

    def _process_reward(self, obs, new_obs, act):
        return 0

    def _process_done(self, obs, new_obs, act):
        return False

    def _process_obs(self, obs):
        return obs.float() # / float(self.size)


class SimpleGridGoal(SimpleGrid):

    def __init__(self, size=2, dim=2):
        super().__init__(size=size, dim=dim)
        self.goal_state = torch.tensor(np.random.choice(self.size, self.dim)).long()

    def _process_reward(self, obs, new_obs, act):
        old_dist = (obs - self.goal_state).abs().sum()
        new_dist = (new_obs - self.goal_state).abs().sum()
        return old_dist - new_dist

    def _process_done(self, obs, new_obs, act):
        return (new_obs - self.goal_state).abs().sum() < 0.001
