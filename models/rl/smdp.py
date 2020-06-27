import logging
import time
from gym import spaces
from .wrapper import Environment


class SMDP(object):
    ''' This class defines an smdp using a decoder function that is assumed to be given. It
        operated on some lower level MDP or semi-MDP through the decoder.
    Arguments:
        base_mdp (required, wrapper.Environment or smdp.SMDP): the lower-level MDP which the SMDP
            will operate on. Must have step method and reset method. If rendering=True then must
            support rendering as well.
        decoder (required, TBD): the decoder model with state size action_dim and output size
            matching the action size of base_mdp. If variable_steps is True, must have a
            termination indicator as output to signal when smdp step ends.
        action_space (required, gym.Space): The space of all admissible actions in the SMDP.
            The output of decoder should be bounded to this set.
        max_steps (required, integer): the maximum number of steps that can be decoded from one
            action. This is here as a failsafe so the decoder doesn't get stuck in some kind of
            loop.
        variable_steps (optional, boolean): if False, the decoder no longer is responsible for
            terminating an action sequence. Instead, the decoder will just be unrolled for
            max_steps steps. (default: True) (This functionality is not currently available)
        rendering (optional, boolean): whether environment rendering is supported.(default: False)
    Attributes:
        base_mdp: the underlying mdp or smdp we turning into a more abstract smdp
        observation_space: the set of all values admissible for observation.
        action_space: the set of all values admissible for action.
        observation: represents the state of the environment at the current time step
        done: when True, episode has terminated
    Methods:
        reset: start a new episode, returns initial obervation
        step: take an action in the current observation and observe the next observation and
            the reward and done variables.
        seed: set the random seed for the underlying environment
    '''
    def __init__(self,
                 base_mdp,
                 autoencoder,
                 gamma,
                 action_space,
                 max_steps,
                 variable_steps=True,
                 rendering=False):

        self.logger = logging.getLogger(__name__)

        if not isinstance(base_mdp, Environment) and not isinstance(base_mdp, SMDP):
            raise ValueError('base_mdp must be an instance of either '
                             'smdp.SMDP or wrapper.Environment')
        self.base_mdp = base_mdp

        self.autoencoder = autoencoder

        if not isinstance(gamma, float):
            raise ValueError('gamma must be float')
        elif gamma < 0 or gamma > 1:
            raise ValueError('gamma must be in the interval [0,1]')
        self.gamma = gamma

        if (not isinstance(action_space, spaces.Box) and not
                isinstance(action_space, spaces.Discrete)):
            raise ValueError('action_space must be an instance of spaces.Box or spaces.Discrete')
        self.action_space = action_space
        self.action_dim = self.action_space.shape

        if not isinstance(max_steps, int):
            raise ValueError('max_steps must be an integer')
        elif max_steps <= 0:
            raise ValueError('max_steps must be an integer > 0')
        self.max_steps = max_steps

        if not isinstance(variable_steps, bool):
            raise ValueError('variable_steps must be boolean')
        self.variable_steps = variable_steps

        if not isinstance(rendering, bool):
            raise ValueError('rendering must be boolean')
        self.rendering = rendering

    def reset(self):
        self.observation = self.base_mdp.reset()
        return self.observation

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        smdp_step_reward = 0
        step_info = {'observations': [],
                     'actions': [],
                     'rewards': [],
                     'Taus': []}

        if self.variable_steps:
            # steps counts how many times a new action is passed to the lower MDP/SMDP
            # Tau tracks how many steps in the basest MDP were taken so that we can compute
            # the value and Q functions
            # decode_done is a boolean indicating whether the decoder has predicted the unroll
            # should terminate and hand back control to the managing agent.
            steps = 0
            Tau = 0
            decode_done = False
            base_ep_done = False

            while not decode_done and not base_ep_done and steps < self.max_steps:
                # One step of the decoder
                # Takes state and action (hidden state of lstm/gru) as input
                # Outputs base action, new action code, termination indicator
                [base_action,
                 next_action,
                 decode_done] = self.autoencoder.decode(encoder_hidden=action,
                                                        observations=self.observation)

                # Take a step in the base_mdp
                (next_observation, base_reward,
                 base_ep_done, base_Tau, _) = self.base_mdp.step(base_action)
                time.sleep(0.05)

                # Add observation, action, and reward to step info
                step_info['observations'].append(self.observation)
                step_info['actions'].append(action)
                step_info['rewards'].append(base_reward)
                step_info['Taus'].append(base_Tau)

                # Update state, hidden state, cumulative reward and time step
                self.observation = next_observation
                action = next_action
                smdp_step_reward += (self.gamma ** Tau) * base_reward
                Tau += base_Tau
                steps += 1
                if decode_done: print('Decode Done')

            return self.observation, smdp_step_reward, base_ep_done, Tau, step_info

        else:
            raise NotImplementedError()
            '''
            for t in range(self.max_steps):

            return self.max_steps
            '''
        return
