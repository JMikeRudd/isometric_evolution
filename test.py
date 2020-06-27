import torch, os
save_dir = 'trained_models/simple_grid_4/agent_embedding_1'
pop_model = torch.load(os.path.join(save_dir, 'pop_model'))
from models.embedding.utils import plt_pca_coords
plt_pca_coords(pop_model.emb_model.model.weight.data.detach().cpu().numpy(), save_dir=save_dir)
