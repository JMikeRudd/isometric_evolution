import os
import math
import logging
import torch
from torch.nn import ELU, ReLU, Sigmoid, Linear, Sequential, Module, Parameter, Softmax, BatchNorm1d
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.utils.data import Dataset, TensorDataset
import numpy as np

USE_CUDA = torch.cuda.is_available()

pi = math.pi

# ==NN Utilities== #
ACT_FNS = ['ELU', 'ReLU', 'Sigmoid']

log_lvls = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.DEBUG,
    'CRITICAL': logging.CRITICAL
}


def linear_stack(layers, activation='ReLU', batch_norm=False):
    assert activation in ACT_FNS
    if activation == 'ELU':
        activation = ELU
    elif activation == 'ReLU':
        activation = ReLU
    elif activation == 'Sigmoid':
        activation = Sigmoid

    net = []
    for i in range(len(layers) - 1):
        if batch_norm:
            net.append(BatchNorm1d(layers[i]))
        net.append(Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            net.append(activation())
    return Sequential(*net)


def make_transformer(d_model, nhead, num_layers):
    encoder_layer = TransformerEncoderLayer(d_model, nhead)
    return TransformerEncoder(encoder_layer, num_layers=num_layers)


class GlobalAttentionHead(Module):

    def __init__(self, inp_dim, out_dim, query_dim=None):

        super().__init__()

        assert isinstance(inp_dim, int) and inp_dim > 0
        self.inp_dim = inp_dim

        assert isinstance(out_dim, int) and out_dim > 0
        self.out_dim = out_dim

        if query_dim is None:
            query_dim = max(inp_dim, out_dim)

        assert isinstance(query_dim, int) and query_dim > 0
        self.query_dim = query_dim

        self.global_query_weights = Parameter(torch.ones(self.query_dim),
                                              requires_grad=True)
        self.key_weights = Linear(self.inp_dim, self.query_dim)
        self.value_weights = Linear(self.inp_dim, self.out_dim)
        # self.sftmx = Softmax()

    def forward(self, embs, lengths):

        # Expects padded sequence with batch first False
        assert issubclass(type(embs), torch.Tensor)
        assert embs.dim() == 3 and embs.size(-1) == self.inp_dim

        # Get batch size
        bs = len(lengths)

        # Create output tensor
        outs = torch.zeros(bs, self.out_dim)
        if embs.device.type == 'cuda':
            outs = outs.cuda()

        for b in range(bs):
            b_seq = embs[:lengths[b].int().item(), b, :]
            b_keys = self.key_weights(b_seq)
            b_vals = self.value_weights(b_seq)

            b_attn_energies = b_keys.matmul(
                self.global_query_weights.unsqueeze(1))
            b_weights = b_attn_energies.exp()
            b_weights = b_weights / b_weights.sum()
            # self.sftmx(b_attn_energies.transpose(0, 1),
            #                       dim=lengths[b].int().item())

            outs[b] = b_vals.transpose(0, 1).matmul(
                b_weights).transpose(0, 1)

        return outs

def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  #assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca, eigen_vals, eigen_vecs


def set_req_grad(model, bool_val):
    assert issubclass(type(model), torch.nn.Module)
    assert isinstance(bool_val, bool)

    for p in model.parameters():
        p.requires_grad = bool_val

def sum_to_1(tens):
    assert issubclass(type(tens), torch.Tensor)
    while tens.dim() > 1:
        tens = tens.sum(dim=-1)
    return tens


def isomap_coords(D, emb_dim, save_dir=None):
    import numpy as np

    device = D.device.type
    n_pts = D.size(0)
    center_mat = (torch.eye(n_pts) - (torch.ones(n_pts, n_pts) / float(n_pts))).to(device)
    B = -0.5 * center_mat.matmul(D.pow(2).to(device)).matmul(center_mat)

    eigvals, eigvecs = np.linalg.eig(B.cpu().numpy())
    coords = np.dot(eigvecs[:,:emb_dim].real, np.diag(np.sqrt(eigvals[:emb_dim].real)))

    if save_dir is not None:
        from matplotlib import pyplot as plt
        # plot coordinates
        plt.scatter(coords[:,0], coords[:,1], s=1)
        plt.savefig(os.path.join(save_dir, 'isomap_coords'))
        plt.close()

        # plot eigenvalues
        eig_val_dim = min(2*emb_dim, len(eigvals))
        plt.bar([i for i in range(eig_val_dim)], np.abs(eigvals[:eig_val_dim]))
        #plt.title('Genotype Embedding Eigenvalues')
        plt.savefig(os.path.join(save_dir, 'isomap_eigvals'))
        plt.close()

    return torch.tensor(coords).float()


def plt_pca_coords(embs, dim=2, save_dir=None, save_name='learned_coords'):
    import numpy as np
    from matplotlib import pyplot as plt

    assert save_dir is not None and os.path.exists(save_dir)

    embs -= np.mean(embs, axis=0)
    embs_pca, eigvals, eigvecs = pca(embs)

    plt.bar([i for i in range(len(eigvals))], eigvals)
    plt.savefig(os.path.join(save_dir, save_name + '_eigenvalues'))
    plt.close()

    for i in range(dim):
        for j in range(i+1, dim):
            plt.scatter(embs_pca[:,i], embs_pca[:, j], s=1)
            plt.savefig(os.path.join(save_dir, save_name + '_{}_{}'.format(i,j)))
            plt.close()



def plot_embs_pca(emb_obs, model, save_dir=None, batch_size=100, n_dims=2, model_name=None):

    import numpy as np
    from matplotlib import pyplot as plt

    assert save_dir is not None and model_name is not None
    eigval_save_path = os.path.join(save_dir, 'eigenvalues')
    emb_save_dir = os.path.join(save_dir, 'PCA_embs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)

    n_samples = emb_obs.size(0)
    emb_dim = model.emb_dim
    embs = torch.zeros((n_samples, emb_dim)).cuda()

    assert isinstance(batch_size, int) and batch_size > 0
    for i in range(0, n_samples, batch_size):
        with torch.no_grad():
            batch_embs = model(emb_obs[i:(i + batch_size)])
        embs[i:(i + batch_size), :] = batch_embs

    embs = embs.cpu().numpy()
    emb_mean = np.mean(embs, 0)
    embs -= emb_mean

    embs_pca, eigvals, eigvecs = pca(embs)
    #embs_pca = embs # must change
    '''
    U, S, V = np.linalg.svd(embs, full_matrices=False, compute_uv=True)
    embs_svd = np.dot(U, np.diag(S))    
    eigvals = S * S / (n_samples - 1)
    '''

    plt.bar([i for i in range(emb_dim)], eigvals)
    plt.title('Embedding Eigenvalues')
    plt.savefig(eigval_save_path)
    plt.close()

    for i in range(n_dims):
        for j in range(i+1, n_dims):
            plt.scatter(embs_pca[:,i], embs_pca[:, j], s=1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('PCA Dimension {}'.format(i+1))
            plt.ylabel('PCA Dimension {}'.format(j+1))
            plt.title('{} 2D Projection of Learned Embeddings'.format(model_name))
            plt.savefig(os.path.join(emb_save_dir, 'emb_{}_{}'.format(i,j)))
            plt.close()


class MixedDataset(Dataset):
    ''' Takes dict of tensors or torch datasets as input. A data item is a
        dict of the ith item of each element of the list
    '''
    def __init__(self, data_dict):
        super().__init__()
        assert isinstance(data_dict, dict)

        lens = []
        self.data_dict = {}
        for k, v in data_dict.items():

            assert (k is not None and
                    (issubclass(type(v), torch.Tensor) or
                     issubclass(type(v), Dataset)))

            if issubclass(type(v), torch.Tensor):
                self.data_dict[k] = TensorDataset(v)
            else:
                self.data_dict[k] = v

            lens.append(len(self.data_dict[k]))

        data_len = int(lens[0])
        assert all([(d_len == data_len) for d_len in lens])
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        ret_dict = {}
        for k, v in self.data_dict.items():
            item = v.__getitem__(index)
            if isinstance(item, list) or isinstance(item, tuple):
                item = item[0]
            ret_dict[k] = item
        return ret_dict

def images_to_gif(img_dir, file_names, out_name, fps=24):
    import os
    import imageio

    assert os.path.exists(img_dir)
    assert isinstance(file_names, list)

    images = []
    for file_name in file_names:
        assert file_name.endswith('.png')
        file_path = os.path.join(img_dir, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(img_dir, '{}.gif'.format(out_name)), images, fps=fps)


def get_gradient_steepest_ascent(embs, signal, norm=True):
    import statsmodels.api as sm
    import numpy as np

    assert issubclass(type(embs), np.ndarray) and issubclass(type(signal), np.ndarray)
    assert embs.shape[0] == signal.shape[0]

    reg_model = sm.OLS(signal, embs).fit()
    # reg_model = LinearRegression()
    # reg_model.fit(embs, signal)
    steepvec = reg_model.params
    signal_res = signal - reg_model.predict(embs)

    p_val = reg_model.f_pvalue
    p_val = p_val if not np.isnan(p_val) else 1.

    if norm:
        steepvec /= np.linalg.norm(steepvec)
    
    return steepvec, signal_res, p_val
