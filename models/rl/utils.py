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
