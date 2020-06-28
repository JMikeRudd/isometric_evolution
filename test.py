import torch, os
save_dir = 'trained_models/simple_grid_4/agent_embedding_2'
pop_model = torch.load(os.path.join(save_dir, 'pop_model'))
from models.embedding.utils import plt_pca_coords
plt_pca_coords(pop_model.emb_model.model.weight.data.detach().cpu().numpy(), save_dir=save_dir)



# Getting best policy from pop_distns and birth model to compare
max_ind = pop_scores.argmax()
best_emb = pop_model.pop_embs[max_ind]
unique_obs = pop_model.metric.unique_obs
pop_model.pop_distns[max_ind]
pop_model.birth_model.get_probs(obs=unique_obs, emb=best_emb.unsqueeze(0).repeat(len(unique_obs), 1))



# Generating progression plots
import torch, os
save_dir = 'trained_models/simple_grid_goal/evolution_momentum_1'
all_scores = torch.load(os.path.join(save_dir, 'all_scores'))
all_embs = torch.load(os.path.join(save_dir, 'all_embs'))
from ea_td import plot_progression

plot_progression(all_embs.numpy(), all_scores.numpy(), save_dir=os.path.join(save_dir, 'training_progression'))
