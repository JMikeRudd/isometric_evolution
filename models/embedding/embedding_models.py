import logging
import torch
from torch.nn import Module, Embedding, GRU, Sequential, Conv2d, ReLU, Parameter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .utils import linear_stack, make_transformer, GlobalAttentionHead
STEP_EMB_MODEL_TYPES = ['Discrete', 'MLP', 'Conv', 'ID', 'NNLookup']
TRAJECTORY_EMB_MODEL_TYPES = ['GRU', 'Transformer']
EMB_MODEL_TYPES = STEP_EMB_MODEL_TYPES + TRAJECTORY_EMB_MODEL_TYPES


class EmbMapping(Module):
    ''' Class for mapping input to embedding. At its heart a torch Module
        with attributes for dimensions of inputs / outputs
        Arguments:
            inp_dim (required):
                Int or tuple of ints representing the input dimension of
                the model. Use inp_dim=-1 to get around validation issues.
            emb_dim (required):
                Int representing size of embedding vector (output of model)
            model:
                torch.nn.Module to map inputs to embeddings
        Methods:
            forward:
                takes input, validates it (dimension), embeds it, and
                validates embedding
            _validate_inp (implemented by subclasses):
                Check that input has right dimensions. Returns fixed
                vector if problem is small, otherwise raises error.
            _validate_emb:
                Check that output has right dimensions. Returns fixed
                vector if problem is small, otherwise raises error.
            _embed (implemented by subclasses):
                process input and return embedding.
    '''
    def __init__(self, inp_dim, emb_dim, model=None):

        super().__init__()

        assert issubclass(type(model), Module) or model is None
        self.model = model

        assert isinstance(emb_dim, int) and emb_dim > 0
        self.emb_dim = emb_dim

        if isinstance(inp_dim, int):
            assert inp_dim > 0 or inp_dim == -1
        else:
            assert isinstance(inp_dim, tuple)
            for d in inp_dim:
                assert isinstance(d, int) and d > 0
        self.inp_dim = inp_dim

        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)

    def forward(self, inp):
        inp = self._validate_inp(inp)
        emb = self._embed(inp)
        val_emb = self._validate_emb(emb)
        return val_emb

    def _validate_inp(self, inp):
        raise NotImplementedError('Implemented by subclasses')

    def _validate_emb(self, emb):
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        assert emb.dim() == 2 and emb.size(1) == self.emb_dim
        return emb

    def _embed(self, inp):
        raise NotImplementedError('Implemented by subclasses')


class DiscreteEmbMapping(EmbMapping):
    ''' Class for when input is discrete (one-hot vector).
    '''
    def __init__(self, inp_dim, emb_dim):
        super().__init__(inp_dim, emb_dim, model=Embedding(inp_dim, emb_dim))

    def _embed(self, inp):
        return self.model(inp)

    def _validate_inp(self, inp):
        # _embed expects long tensor of indices with dim (batch_size, 1)

        # Check that no inputs are negative
        assert (inp >= 0).all() and (inp <= self.inp_dim).all()
        # If we receive 1-D tensor we must check whether it is:
        # 1) A tensor of indices that must be unsqueezed to have batched shape
        # 2) A single one-hot vector that must be converted to a single ind
        '''
        if inp.dim() == 1:
            if ((inp > 1).any() or inp.size(0) != self.inp_dim or
                    inp.sum() != 1):
                return inp.unsqueeze(1).long()
            else:
                _, inp = inp.max(dim=-1)
                return inp.unsqueeze(0)

        assert inp.dim() == 2
        # Now we check for one-hot vectors and indices out of range
        if (inp.size(1) == self.inp_dim and (inp <= 1).all() and
                (inp.sum(1) == 1).all()):
            _, inp = inp.max(dim=-1)
        else:
            assert inp.size(1) == 1
            assert (inp <= self.inp_dim).all()
        '''
        return inp


class MLPEmbMapping(EmbMapping):
    ''' Embedding is MLP applied to continuous variable
        Arguments:
            layer sizes:
                List of integers. First element is inp size, last is emb size.
                All others are the sizes of any hidden layers
            act_fn:
                Activation function of hidden layers. Last layer has no act_fn
    '''
    def __init__(self, layer_sizes, act_fn='ReLU', batch_norm=False):

        super().__init__(**MLPEmbMapping._init_MLP(layer_sizes, act_fn, batch_norm))

    @staticmethod
    def _init_MLP(layer_sizes, act_fn='ReLU', batch_norm=False):
        assert isinstance(layer_sizes, list)
        assert len(layer_sizes) >= 2
        for ls in layer_sizes:
            assert isinstance(ls, int) and ls > 0
        inp_dim = layer_sizes[0]
        emb_dim = layer_sizes[-1]

        model = linear_stack(layer_sizes, act_fn, batch_norm)

        return {'inp_dim': inp_dim, 'emb_dim': emb_dim, 'model': model}

    def _embed(self, inp):
        return self.model(inp)

    def _validate_inp(self, inp):
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            assert inp.dim() == 2 and inp.size(1) == self.inp_dim
        return inp


class ConvEmbMapping(EmbMapping):
    ''' Embedding is a convnet applied to image-style input
        Arguments:
            layer sizes:
                List of integers. First element is inp size, last is emb size.
                All others are the sizes of any hidden layers
            act_fn:
                Activation function of hidden layers. Last layer has no act_fn
    '''
    def __init__(self, image_dim, filter_sizes, conv_size=3, act_fn='ReLU'):

        super().__init__(**ConvEmbMapping._init_net(image_dim, filter_sizes, conv_size=conv_size, act_fn=act_fn))

    @staticmethod
    def _init_net(image_dim, filter_sizes, conv_size=3, dilation=2, act_fn='ReLU'):
        assert isinstance(image_dim, tuple) and len(image_dim) == 3
        inp_dim = image_dim
        assert 0.5 * (conv_size - 1) == int(0.5 * (conv_size - 1))
        pad_dim = int(0.5 * dilation * (conv_size - 1))

        assert isinstance(filter_sizes, list)
        assert len(filter_sizes) >= 1
        assert act_fn in  ['ReLU']
        in_channels = image_dim[0]
        model_blocks = []
        for ls in filter_sizes[:-1]:
            assert isinstance(ls, int) and ls > 0
            model_blocks += [Conv2d(in_channels, ls, conv_size, padding=(pad_dim, pad_dim), dilation=dilation), ReLU()]
            in_channels = ls

        # Add last block to 
        emb_dim = filter_sizes[-1]
        model_blocks += [Conv2d(in_channels, emb_dim, (image_dim[1], image_dim[2])), ReLU()]

        # Add some fully-connected layers
        model_blocks += [Conv2d(emb_dim, 2 * emb_dim, (1, 1)), ReLU(), Conv2d(2 * emb_dim, emb_dim, (1, 1))]

        # Make model from blocks
        model = Sequential(*model_blocks)

        return {'inp_dim': inp_dim, 'emb_dim': emb_dim, 'model': model}

    def _embed(self, inp):
        return self.model(inp).squeeze(-1).squeeze(-1)

    def _validate_inp(self, inp):
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            assert inp.dim() == 2 and inp.size(1) == self.inp_dim
        return inp


class IDEmbMapping(EmbMapping):
    ''' Embedding is exactly the input. This is useful for trajectory emb models.
        Arguments:
            layer sizes:
                List of integers. First element is inp size, last is emb size.
                All others are the sizes of any hidden layers
            act_fn:
                Activation function of hidden layers. Last layer has no act_fn
    '''
    def __init__(self, inp_dim):
        super().__init__(inp_dim=inp_dim, emb_dim=inp_dim, model=None)

    def _embed(self, inp):
        return inp

    def _validate_inp(self, inp):
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            assert inp.dim() == 2 and inp.size(1) == self.inp_dim
        return inp


class NNLookupEmbMapping(EmbMapping):
    ''' Class for when input is discrete (one-hot vector).
    '''
    def __init__(self, inp_tensor, emb_tensor):
        super().__init__(**NNLookupEmbMapping._init_NNLookup(inp_tensor, emb_tensor))

        self.inp_tensor = inp_tensor
        self.emb_tensor = emb_tensor

        self.model.weight.data = emb_tensor.clone()

    @staticmethod
    def _init_NNLookup(inp_tensor, emb_tensor):
        assert issubclass(type(inp_tensor), torch.Tensor) and issubclass(type(emb_tensor), torch.Tensor)
        assert inp_tensor.size(0) == emb_tensor.size(0)
        
        inp_dim = inp_tensor.size(0)

        assert emb_tensor.dim() == 2
        emb_dim = emb_tensor.size(1)
        model = Embedding(inp_dim, emb_dim)
        return {'inp_dim': inp_dim, 'emb_dim': emb_dim, 'model': model}

    def _embed(self, inp):
        return self.model(inp)

    def _validate_inp(self, inp):
        # Convert inputs to indices with NN lookup
        device = self.inp_tensor.device.type
        assert inp.size()[1:] == self.inp_tensor.size()[1:]

        inp_inds = torch.zeros(inp.size(0)).long().to(device)
        for i in range(inp.size(0)):
            inp_inds[i] = (inp[i]-self.inp_tensor).pow(2).view(self.inp_tensor.size(0),-1).sum(1).argmin()

        # Check that no inputs are negative
        assert (inp_inds >= 0).all() and (inp_inds <= self.inp_dim).all()
        return inp_inds.to(device)


class MixedEmbMapping(EmbMapping):
    ''' Takes multiple EmbMapping models and processes them into a single
        embedding with dim emb_dim
        Arguments:
            emb_model_dict (dict, required):
                Dictionary of EmbMapping objects. Dictionary names should
                correspond to keys found in input.
            emb_dim:
                Positive integer representing embedding size.
    '''
    def __init__(self, emb_model_dict, emb_dim, comb_model=None):
        super().__init__(inp_dim=-1, emb_dim=emb_dim, model=None)
        assert isinstance(emb_model_dict, dict)
        for v, m in emb_model_dict.items():
            assert issubclass(type(m), EmbMapping)
            self.__setattr__(v, m)

        self.emb_model_dict = emb_model_dict
        cat_dim = int(
            torch.tensor([m.emb_dim for m in
                          self.emb_model_dict.values()]).sum().item())
        if comb_model is None:
            if cat_dim == emb_dim:
                self.comb_model = IDEmbMapping(cat_dim)
            else:
                self.comb_model = MLPEmbMapping([cat_dim, emb_dim])
        else:
            assert issubclass(type(comb_model), EmbMapping)
            assert (comb_model.inp_dim == cat_dim and
                    comb_model.emb_dim == emb_dim)
            self.comb_model = comb_model

        self.keys = sorted(self.emb_model_dict.keys())

    def _embed(self, inp):
        emb_list = []
        for k in self.keys:
            model_k = self.emb_model_dict[k]

            # Trajectory embeddings and mixed embeddings expect dict,
            # so give them everything
            if (issubclass(type(model_k), MixedEmbMapping) or
                    issubclass(type(model_k), TrajectoryEmbMapping)):
                k_emb = model_k(inp)
            else:
                k_emb = model_k(inp[k])

            emb_list.append(k_emb)

        mid = torch.cat(emb_list, dim=-1)
        return self.comb_model(mid)

    def _validate_inp(self, inp):
        assert isinstance(inp, dict)
        for k in self.keys:
            assert k in inp.keys()
        return inp


class Scaled(EmbMapping):

    def __init__(self, emb_mapping):
        assert issubclass(type(emb_mapping), EmbMapping)
        super().__init__(emb_mapping.inp_dim, emb_mapping.emb_dim, emb_mapping.model)
        self.emb_mapping = emb_mapping

        self.scale_param = Parameter(torch.ones(emb_mapping.emb_dim), requires_grad=True)

    def _embed(self, inp):
        return self.scale_param * self.emb_mapping._embed(inp)

    def _validate_inp(self, inp):
        return self.emb_mapping._validate_inp(inp)


class TrajectoryEmbMapping(EmbMapping):
    ''' Class for embedding trajectories of observations. These can vary in length
        but will always have a temporal order.
        Arguments:
            emb_dim (required):
                Positive integer representing embedding size.
            obs_model (required):
                MixedEmbMapping to process each observation.
            temporal_model (required):
                Any trajectory embedding model must have a component that
                aggregates multiple time steps into a single vector. This is
                expected to be passed by subclasses.
            max_len (int, optional):
                Maximum trajectory length to allow. If -1 allows any length.
                Default -1.
    '''
    def __init__(self, emb_dim, obs_model, temporal_model, max_len=-1):
        super().__init__(**TrajectoryEmbMapping._init_TrajEmb(
            emb_dim, obs_model, temporal_model, max_len))

        self.max_len = max_len

        self.obs_model = obs_model
        self.temporal_model = temporal_model

    @staticmethod
    def _init_TrajEmb(emb_dim, obs_model, temporal_model, max_len):

        assert isinstance(max_len, int) and max_len == -1 or max_len > 0

        # obs_model must be MixedEmbMapping because we expect to receive
        # input as dictionary. If input is single tensor a MixedEmbMapping
        # can still be declared (with a single component)
        assert issubclass(type(obs_model), MixedEmbMapping)
        assert (issubclass(type(temporal_model), Module) or
                temporal_model is None)

        return {'inp_dim': -1, 'emb_dim': emb_dim, 'model': None}

    def _embed(self, inp):
        proc_obs = self._process_obs(inp)
        emb = self._aggregate_steps(step_embs=proc_obs, lengths=inp['lengths'])
        return emb

    def _process_obs(self, obs):

        bs = len(obs['lengths'])

        # Create dicts to feed to obs model for each trajector in batch
        inp_dicts = [{k: obs[k][i] for k in obs.keys() if k != 'lengths'}
                     for i in range(bs)]

        # For each trajectory in the batch process the observations into
        # a single embedding vector for each time step
        step_embs = [self.obs_model(inp_dicts[i]) for i in range(bs)]

        return step_embs

    def _aggregate_steps(self, step_embs, lengths=None):
        raise NotImplementedError('Implemented by subclasses')

    def _validate_inp(self, inp):
        # Expects to receive dict with entries of:
        # 1) 'length': tensor sequence lengths [l_1,...,l_b] in batch.
        # 2) <var_name>: list of tensors of shape (li,*)

        assert isinstance(inp, dict)
        assert 'lengths' in inp.keys()

        bs = inp['lengths'].size(0)  # len(inp['lengths'])

        # Check that all dict entries have same batch size
        for k in inp.keys():
            assert len(inp[k]) == bs

            # Check all inputs are tensors with same seq length as in lengths
            '''
            if k != 'lengths':
                for i in range(bs):
                    assert issubclass(type(inp[k][i]), torch.Tensor)
                    assert inp[k][i].size(0) == inp['lengths'][i]
            '''

        return inp


class GRUTrajectoryEmbMapping(TrajectoryEmbMapping):
    ''' TrajectoryEmbMapping where temporal model is a GRU
    '''
    def __init__(self, emb_dim, obs_model, n_layers=1, max_len=-1):
        super().__init__(
            emb_dim, obs_model,
            temporal_model=GRUTrajectoryEmbMapping._init_GRU(emb_dim,
                                                             obs_model,
                                                             n_layers),
            max_len=max_len)

    def _init_GRU(emb_dim, obs_model, n_layers):
        assert issubclass(type(obs_model), EmbMapping)
        temporal_model = GRU(input_size=obs_model.emb_dim,
                             hidden_size=emb_dim,
                             num_layers=n_layers,
                             batch_first=True)
        return temporal_model

    def _aggregate_steps(self, step_embs, lengths):

        # Expects list of tensors for each trajectory in batch
        padded_step_embs = pad_sequence(step_embs, batch_first=True)

        packed_step_embs = pack_padded_sequence(padded_step_embs, lengths,
                                                batch_first=True,
                                                enforce_sorted=False)
        _, traj_emb = self.temporal_model(packed_step_embs)

        return traj_emb.squeeze(0)


class TransformerTrajectoryEmbMapping(TrajectoryEmbMapping):
    ''' TrajectoryEmbMapping where temporal model is a transformer
    '''
    def __init__(self, emb_dim, obs_model, n_layers=1, nhead=8, max_len=-1):
        super().__init__(emb_dim, obs_model,
                         temporal_model=None, max_len=max_len)

        assert issubclass(type(obs_model), EmbMapping)
        assert isinstance(n_layers, int) and n_layers > 0
        assert isinstance(nhead, int) and nhead > 0
        self.base = make_transformer(d_model=obs_model.emb_dim,
                                     nhead=nhead,
                                     num_layers=n_layers)
        self.aggregator = GlobalAttentionHead(inp_dim=obs_model.emb_dim,
                                              out_dim=emb_dim)

    def _aggregate_steps(self, step_embs, lengths):

        if hasattr(self, 'logger'):
            self.logger.warning('Need to implement position embeddings')
        # Expects list of tensors for each trajectory in batch
        padded_step_embs = pad_sequence(step_embs, batch_first=False)

        bs = lengths.size(0)

        # Seq mask is 1 where we should ignore
        seq_masks = (1 - pad_sequence([torch.ones((1, lengths[i].int()))
                                       for i in range(bs)],
                                      batch_first=False)).squeeze(0).bool()
        if padded_step_embs.device.type == 'cuda':
            seq_masks = seq_masks.cuda()

        transformed_step_embs = self.base(padded_step_embs,
                                          src_key_padding_mask=seq_masks)

        traj_emb = self.aggregator(transformed_step_embs, lengths=lengths)

        return traj_emb.squeeze(0)


def get_emb_model(
        inp_summary, emb_dim, n_layers=1, hidden_size=None):

    model_dict = {}
    for k in inp_summary.keys():
        assert isinstance(inp_summary[k], dict)
        assert ('dim' in inp_summary[k].keys() and
                'type' in inp_summary[k].keys())
        assert inp_summary[k]['type'] in EMB_MODEL_TYPES

        inp_dim_k = inp_summary[k]['dim']
        emb_dim_k = inp_summary[k].get('emb_dim', int(0.5 * inp_dim_k + 0.5))

        if inp_summary[k]['type'] in STEP_EMB_MODEL_TYPES:
            if inp_summary[k]['type'] == 'Discrete':
                model_dict[k] = DiscreteEmbMapping(inp_dim_k, emb_dim_k)
            elif inp_summary[k]['type'] == 'MLP':
                layers_k = [inp_dim_k] + [hidden_size] * n_layers + [emb_dim_k]
                model_dict[k] = MLPEmbMapping(layers_k)
            elif inp_summary[k]['type'] == 'ID':
                model_dict[k] = IDEmbMapping(inp_dim_k)

        elif inp_summary[k]['type'] in TRAJECTORY_EMB_MODEL_TYPES:
            assert 'obs_model' in inp_summary[k].keys()

            if issubclass(type(inp_summary[k]['obs_model']), EmbMapping):
                obs_model_k = inp_summary[k]['obs_model']

            else:
                assert (isinstance(inp_summary[k]['obs_model'], str) and
                        inp_summary[k]['obs_model'] in STEP_EMB_MODEL_TYPES)
                obs_model_k_summary = {
                    k: {'type': inp_summary[k]['obs_model'],
                        'dim': inp_dim_k}}

                assert hidden_size is not None
                assert isinstance(hidden_size, int) and hidden_size > 0
                obs_emb_dim = hidden_size

                if inp_summary[k]['type'] == 'Transformer':
                    assert obs_emb_dim % inp_summary[k].get('nhead', 8) == 0

                obs_model_k = get_emb_model(obs_model_k_summary, obs_emb_dim,
                                            n_layers, hidden_size)

            max_len_k = inp_summary[k].get('max_len', -1)

            if inp_summary[k]['type'] == 'GRU':
                model_dict[k] = GRUTrajectoryEmbMapping(emb_dim_k, obs_model_k,
                                                        n_layers=n_layers,
                                                        max_len=max_len_k)
            elif inp_summary[k]['type'] == 'Transformer':
                nhead_k = inp_summary[k].get('nhead', 8)
                model_dict[k] = TransformerTrajectoryEmbMapping(
                    emb_dim_k, obs_model_k,
                    n_layers=n_layers, nhead=nhead_k, max_len=max_len_k)

    return MixedEmbMapping(model_dict, emb_dim)
