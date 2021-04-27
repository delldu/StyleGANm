import torch
import pdb

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, emb_size, batch_size, n_batches):
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_batches = n_batches
		# pdb.set_trace()
		# emb_size = 512
		# batch_size = 2
		# n_batches = 10000

    def __len__(self):
        return self.batch_size * self.n_batches

    def __getitem__(self, idx):
        return {"noise": torch.randn(self.emb_size)}
