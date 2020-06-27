import logging
import time
import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, _utils
import numpy as np


class Buffer(Dataset):
    '''
    Collect a rolling history of observations with operations for adding
    new ones and support for dataloader sampling. Assumes items are dicts of
    torch tensors.
    '''
    def __init__(self, mdp, size, batch_size=64, transfer_threshold=1000):
        assert isinstance(size, int) and size > 0
        self.buffer = {}
        self.off_device_buffer = {}
        self.size = size

        assert isinstance(batch_size, int) and batch_size > 0
        self.batch_size = batch_size

        assert isinstance(transfer_threshold, int) and transfer_threshold > 0
        self.transfer_threshold = transfer_threshold

        self.keys = None
        self.device = 'cpu'

        #self.sample_dist = 

        self.collate_fn = _utils.collate.default_collate

    def __len__(self):
        return self._dict_len(self.buffer)

    def _dict_len(self, in_dict):
        dict_keys = [k for k in in_dict.keys() if k in self.keys]
        
        if len(dict_keys) == 0:
            return 0
        else:
            return len(in_dict[dict_keys[0]])

    def __getitem__(self, idx):
        assert len(self) > 0
        samp_min_ind = max(len(self) - 10, 0)
        new_idx = min(idx % len(self), samp_min_ind)
        return {k: self.buffer[k][new_idx] for k in self.keys}

    def cuda(self):
        self.to('cuda')

    def to(self, device):
        assert device in ['cpu', 'cuda']
        self.device = device
        self.off_device = [d for d in ['cpu', 'cuda'] if d != self.device][0]
        if len(self) > 0:
            for k in self.keys:
                self.buffer[k] = self.buffer[k].to(self.device)

    def add_item(self, new_item):
        assert isinstance(new_item, dict)
        assert all([issubclass(type(v), torch.Tensor) for (k,v) in new_item.items()])
        
        if len(self) == 0:
            self.keys = list(new_item.keys())
            for k in self.keys:
                self.buffer[k] = new_item[k].unsqueeze(0).to(self.device)
        else:
            new_device = new_item[self.keys[0]].device.type
            assert all([k in new_item.keys() for k in self.keys])
            if new_device == self.device:
                for k in self.keys:
                    if len(self) >= self.size:
                        self.buffer[k] = torch.cat([new_item[k].unsqueeze(0),
                                                    self.buffer[k][:-1]])
                    else:
                        self.buffer[k] = torch.cat([new_item[k].unsqueeze(0),
                                                    self.buffer[k]])
            else:
                if len(self.off_device_buffer) == 0:
                    for k in self.keys:
                        self.off_device_buffer[k] = new_item[k].unsqueeze(0)
                else:
                    for k in self.keys:
                        self.off_device_buffer[k] = torch.cat([new_item[k].unsqueeze(0),
                                                    self.off_device_buffer[k]])

        if self._dict_len(self.off_device_buffer) > self.transfer_threshold:
            self._batch_transfer()

    def _batch_transfer(self):
        '''
        Moving data to GPU is very slow but necessary. It can apparently be sped up by batching transfers.
        This function will take any steps held on the CPU and add them to the GPU queue
        '''
        for k in self.keys:
            self.buffer[k] = torch.cat([self.off_device_buffer[k].to(self.device), self.buffer[k]])
            if len(self.buffer[k]) >= self.size:
                self.buffer[k] = self.buffer[k][:self.size]

        self.off_device_buffer = {}


    def sample(self, n=None):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        
        if n is not None:
            assert isinstance(n, int) and n > 0
        else:
            n = self.batch_size

        draw_start = time.time()
        #samp_idxs = Categorical(probs=torch.ones(len(self)) / float(len(self))).sample([n])
        samp_idxs = list(np.random.choice(len(self), n, replace=True))
        samps = [self.__getitem__(i) for i in samp_idxs]
        logger.debug('within sample: {}'.format(time.time()-draw_start))
        dict_start = time.time()
        #ret_dict = {k: self.buffer[k][samp_idxs] for k in self.keys}
        ret_dict = self.collate_fn(samps)
        logger.debug('dict: {}'.format(time.time()-dict_start))
        return ret_dict

    def empty(self):
        self.buffer = {}
        self.keys = None
