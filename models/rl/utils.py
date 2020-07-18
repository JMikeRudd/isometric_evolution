import torch

DEFAULT_EPSILON = 0.25
DEFAULT_GAMMA = 0.99

def onehot(idx, dim):
    assert isinstance(idx, int) and isinstance(dim, int)
    assert 0 <= idx <= dim
    ohvec = torch.zeros(dim)
    ohvec[idx] = 1
    return ohvec

def to_torch(step):
    return torch.tensor(inp)

def rec_to(inp, device):
    if type(inp) not in [dict, list, tuple]:
        assert issubclass(type(inp), torch.Tensor)
        return inp.to(device)
    elif isinstance(inp, list):
        return [rec_to(i, device) for i in inp]
    elif isinstance(inp, tuple):
        return tuple([rec_to(i, device) for i in inp])
    elif isinstance(inp, dict):
        return {k: rec_to(v, device) for (k, v) in inp.items()}
    else:
        raise ValueError('{} is not a supported data structure'.format(type(inp)))


def plt_exp_rewards(all_rewards, lperc=25., uperc=75., hist_len=1, save_dir=None):

    import os
    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert lperc < 50. < uperc

    from matplotlib import pyplot as plt
    import numpy as np

    # Smooth rewards
    smooth_rewards = np.zeros((all_rewards.size(0), all_rewards.size(1) - hist_len + 1))
    for i in range(all_rewards.size(0)):
        smooth_rewards[i,:] += np.convolve(all_rewards[i,:].cpu().numpy(), np.ones((hist_len,))/hist_len, mode='valid')

    smooth_rewards = torch.tensor(smooth_rewards).float()

    def percentile(t: torch.tensor, q: float):
        """
        Not My Code. See https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
        Return the ``q``-th percentile of the flattened input tensor's data.
        
        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.
           
        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values
        return result.unsqueeze(0)

    x = [int(i) for i in range(smooth_rewards.size(1))]
    rew_medians = torch.cat([percentile(smooth_rewards[:, i], 50.) for i in x]).cpu().numpy()
    rew_lowers = torch.cat([percentile(smooth_rewards[:, i], lperc) for i in x]).cpu().numpy()
    rew_uppers = torch.cat([percentile(smooth_rewards[:, i], uperc) for i in x]).cpu().numpy()

    plt.clf()
    plt.fill_between(x, rew_lowers, rew_uppers, color='pink')
    plt.plot(x, rew_medians, color='red')
    plt.title('Generation Average Scores')
    plt.xlabel('Generation')
    plt.savefig(os.path.join(save_dir, 'experiment_scores'))
    plt.close()