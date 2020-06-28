from tqdm import tqdm
import os, logging

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import kl_div, softmax, relu
from torch.distributions import Categorical, Beta
    
from .embedding.embedding_space import EmbeddingSpace, get_embedding_space
from .embedding.embedding_models import DiscreteEmbMapping, NNLookupEmbMapping
from .embedding.metrics import AgentMetric, TableMetric
from .embedding.isometric_embedding import IsometricEmbedding, train_isometric_embedding

from .agent import GoalDirectedStochPolicy

class PopModel(torch.nn.Module):
    """docstring for PopModel"""
    def __init__(self, emb_space, metric, birth_model):
        super(PopModel, self).__init__()
        
        assert issubclass(type(emb_space), EmbeddingSpace)

        self._set_emb_space(emb_space)

        assert issubclass(type(metric), AgentMetric)
        self.metric = metric
        self.n_states = metric.n_states

        assert issubclass(type(birth_model), GoalDirectedStochPolicy)
        self.birth_model = birth_model
        self.act_dim = birth_model.act_dim

    def _set_emb_space(self, emb_space):
        self.emb_space = emb_space
        self.emb_model = self.emb_space.mapping
        self.emb_dim = self.emb_model.emb_dim
        self.pop_size = self.emb_model.inp_dim
        self.pop_embs = self.emb_model.model.weight.data


    def init_embs(self, lap_k=0.03, epochs=10000, bs=100, optim=None, save_dir=None):

        assert save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # First generation will be randomly selected distributions over actions
        '''
        pop_distns = torch.zeros(self.pop_size * self.n_states, self.act_dim)
        for i in range(len(pop_distns)):
            pop_distns[i, torch.randint(4)] += 1
        '''
        
        #self.dist_lookup = NNLookupEmbMapping(inp_tensor=self.metric.unique_obs, emb_tensor=self.pop_distns)
        #pop_distns = torch.randint(low=0, high=1000, size=(self.pop_size, self.n_states, self.act_dim)).float()
        #pop_distns += lap_k * torch.ones((self.pop_size, self.n_states, self.act_dim))
        #pop_distns = softmax(pop_distns, dim=-1)
        
        # Get distance matrix
        D_path, dist_path = os.path.join(save_dir, 'D'), os.path.join(save_dir, 'pop_distns')
        if os.path.exists(D_path) and os.path.exists(dist_path):
            self.D = torch.load(D_path)
            self.pop_distns = torch.load(dist_path)

            assert self.D.size(0) == self.D.size(1) == self.pop_distns.size(0)
        else:
            # Pop distns will be close to deterministic
            act_samples = torch.randint(0, self.act_dim, [self.pop_size, self.n_states])
            pop_distns = torch.nn.functional.one_hot(act_samples, self.act_dim) + lap_k
            pop_distns /= pop_distns.sum(dim=-1, keepdim=True)

            # Mix together so we have continuum of distributions but with weight close to 0 and 1
            mix_weights = Beta(torch.tensor([1.]), torch.tensor([1.])).sample([self.pop_size]).unsqueeze(-1)
            mix_candidates = torch.randint(0, self.pop_size, [self.pop_size, 2])
            self.pop_distns = mix_weights * pop_distns[mix_candidates[:,0]] + (1- mix_weights) * pop_distns[mix_candidates[:,1]]

            torch.save(self.pop_distns, dist_path)
    
            self.D = self._get_dist_matrix(self.pop_distns)
            torch.save(self.D, D_path)

        # Construct isometric embedding
        device = [p for p in self.emb_model.parameters()][0].device.type
        if self.emb_space.type == 'euclidean':
            assert issubclass(type(self.emb_model), DiscreteEmbMapping)

            from .embedding.utils import isomap_coords
            coords = isomap_coords(self.D, self.emb_dim, save_dir=save_dir).to(device)

            # If dimension is higher than needed some columns will be nan, replace with 0
            coords[coords != coords] = 0

            # Try to replace all attributes with new weights
            self.emb_model.model.weight.data = coords
            new_emb_space = get_embedding_space(self.emb_space.type, self.emb_model)

            self._set_emb_space(new_emb_space)

        else:
            init_metric = TableMetric(self.D.to(device))
            isom_embedding = IsometricEmbedding(self.emb_space, init_metric).to(device)

            data_loader = DataLoader(TensorDataset(torch.arange(self.pop_size).to(device)), batch_size=bs, shuffle=True)

            train_isometric_embedding(isom_embedding, epochs=epochs, data_loader=data_loader, optim=optim,
                                      print_every=1, save_every=1000, save_dir=save_dir)

        torch.save(self.to(device), os.path.join(save_dir, 'pop_model'))

        return self.pop_embs, self.pop_distns.to(device)

    def _get_dist_matrix(self, distns):

        # Compute distances between each policy
        D = torch.zeros(distns.size(0), distns.size(0))
        for i in tqdm(range(distns.size(0))):
            for j in range(i):
                D[i, j] += self.metric.dist_from_pols(distns[i], distns[j])
                D[j, i] += D[i, j]

        D /= D.max()
        
        assert (D > -0.001).all()

        return relu(D)


def train_birth_model(birth_model, embs, obs, pop_distns, optim, epochs=100, bs=32,
                      save_dir=None, save_every=100, print_every=1):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct data loader
    assert embs.size(0) == pop_distns.size(0)
    from .embedding.utils import MixedDataset

    device = embs.device.type
    dataset = MixedDataset({
        'obs': torch.arange(len(obs)).repeat(len(pop_distns)).to(device),
        'id': torch.arange(len(pop_distns)).repeat(len(obs)).to(device)})
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for e in tqdm(range(epochs)):
        epoch_loss = train_birth_model_epoch(birth_model, data_loader, embs=embs, obs=obs, pop_distns=pop_distns, optim=optim)

        # Record the epoch loss
        if e % print_every == 0:
            _log_losses(e, epoch_loss, logger)

        if e % save_every == 0 or e == epochs - 1:
            # save the model
            torch.save(birth_model,
                       os.path.join(save_dir, 'birth_model'))

def train_birth_model_epoch(birth_model, data_loader, embs, obs, pop_distns, optim):
    epoch_losses = 0
    N = len(data_loader)

    for (_, batch) in enumerate(data_loader):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        if not isinstance(batch, dict):
            assert issubclass(type(batch), torch.Tensor)
        else:
            for (_, v) in batch.items():
                assert issubclass(type(v), torch.Tensor)

        batch_embs = embs[batch['id']]
        batch_obs = obs[batch['obs']]
        batch_distns = pop_distns[batch['id']].gather(1,batch['obs'].unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze(1)

        batch_loss = train_birth_model_batch(birth_model,
                                             embs=batch_embs, obs=batch_obs,
                                             distns=batch_distns, optim=optim)

        epoch_losses += batch_loss / N

    return epoch_losses

def train_birth_model_batch(birth_model, embs, obs, distns, optim):

    optim.zero_grad()

    pred_dist = birth_model.get_probs(obs=obs, emb=embs)

    loss = kl_div(pred_dist, distns, reduction='batchmean')
    loss.backward()
    optim.step()

    return loss.item()
    

def _log_losses(epoch, epoch_loss, logger):

    log_msg = 'Epoch {}: {}'.format(epoch, epoch_loss)
    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)


class GoalPolicyWrapper(torch.nn.Module):
    """ docstring for GoalPolicyWrapper
        Goal policy expects an embedding as input but sometimes we want
        it to behave like a normal policy given a fixed embedding
    """
    def __init__(self, goal_policy, emb):
        super(GoalPolicyWrapper, self).__init__()
        
        assert issubclass(type(goal_policy), GoalDirectedStochPolicy)
        self.goal_policy = goal_policy

        assert issubclass(type(emb), torch.Tensor)
        self.emb = emb

    def get_probs(self, obs, **kwargs):
        return self.goal_policy.get_probs(obs, emb=self.emb, **kwargs)

    def get_action(self, **kwargs):
        return self.goal_policy.get_action(emb=self.emb, **kwargs)
        
