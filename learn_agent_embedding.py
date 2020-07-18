'''
### Pseudocode ###

Initialize MDP to learn on

Get unique obs, distance metric, emb model etc
Initialize population of solutions

Pass objects to learning routine

Learning routine:
    (Must simultaneously learn policy, value function, and embedding)
    1. Learn population embedding model and reconstruction model (?)
    2. Evaluate population fitness scores
    3. Select candidates for reproduction and generate offspring
    
'''

import argparse
import logging
import os
from copy import copy, deepcopy
from tqdm import tqdm
import time

import torch
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal

from models.embedding.embedding_space import (
    get_embedding_space,
    EMBEDDING_SPACES)
from models.embedding.embedding_models import (
    MLPEmbMapping, ConvEmbMapping, DiscreteEmbMapping,
    IDEmbMapping, MixedEmbMapping, Scaled)
from models.embedding.metrics import JSDAgentMetric, TVAgentMetric
from models.embedding.distns import get_policy_distn
from models.embedding.utils import set_req_grad, pca, sum_to_1

from models.rl.envs import ENVS, get_mdp
from models.rl.utils import rec_to, DEFAULT_EPSILON, DEFAULT_GAMMA
from models.rl.buffer import Buffer
from models.pop_model import PopModel, train_birth_model
from models.agent import GoalDirectedStochPolicy
from models.rl.goal_directed_rl import GoalMDPWrapper, get_goal_mdp_wrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(env_name, 
         emb_space_type='euclidean', emb_dim=10, metric_type='TV',
         state_emb_model_type='MLP', birth_model_hs=None, birth_lr=0.001,
         state_emb_layers=1, state_emb_hidden_size=12,
         unique_samples=2000, isom_epochs=10000, isom_bs=32, isom_lr=0.001,
         pop_size=100, opt_cls='Adam', birth_epochs=100, birth_bs=32,
         model_dir=None, name=None, save_every=5, print_every=1, seed=None, plt=False):

    if seed is not None:
        torch.manual_seed(seed)

    if model_dir is None:
        model_dir = os.path.join(os.getcwd(), 'trained_models')

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    assert name is not None and isinstance(name, str)
    assert env_name in ENVS

    save_dir = os.path.join(model_dir, env_name, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all arguments of the current training run for inspection later if desired
    logging.basicConfig(filename=os.path.join(save_dir, 'train.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    train_args = copy(locals())
    logger.info('Train Arguments:')
    kmaxlen = max([len(k) for k in train_args.keys()])
    formatStr = '\t{:<' + str(kmaxlen) + '} {}'
    for k in sorted(train_args.keys()):
        logger.info(formatStr.format(k, train_args[k]))
    logger.info('Device: {}'.format(device))

    # Get MDP to train on
    env_path = os.path.join(save_dir, 'env')
    env, obs_type, obs_dim, act_dim = get_mdp(env_name)
    # if not os.path.exists(env_path):
    #     env, obs_type, obs_dim, act_dim = get_goal_mdp_wrapper(env_name, n_tasks=1)
    #     env.save(env_path)
    # else:
    #     _, obs_type, obs_dim, act_dim = get_goal_mdp_wrapper(env_name, n_tasks=1)
    #     env = GoalMDPWrapper.load(env_path)

    # Distance between two policies is defined on unique obs
    # If unique obs and embeddings already exist load them, else generate some
    unique_obs_path = os.path.join(save_dir, 'unique_obs')
    if os.path.exists(unique_obs_path):
        unique_obs = torch.load(unique_obs_path)
    else:
        unique_obs = env.reset().clone().unsqueeze(0).to(device)
        for i in range(unique_samples):
            new_obs = env.reset().to(device)
            # if new_obs not in sampled obs, must add
            if sum_to_1((unique_obs - new_obs.unsqueeze(0)).abs()).min() > 0.:
                unique_obs = torch.cat([unique_obs, new_obs.unsqueeze(0)])

    # Get distance metric
    if metric_type == 'TV':
        metric = TVAgentMetric(unique_obs)
    elif metric_type == 'JSD':
        metric = JSDAgentMetric(unique_obs)

    # Get embedding model
    emb_model = DiscreteEmbMapping(pop_size, emb_dim)

    # Get embedding space
    emb_space = get_embedding_space(emb_space_type, emb_model)

    # Get Birth Model (model to map embedding to policy over actions)
    if state_emb_model_type == 'MLP':
        assert obs_type == 'cont' and isinstance(obs_dim, int) and obs_dim > 0
        layers = [obs_dim] + [state_emb_hidden_size] * state_emb_layers + [emb_dim]
        state_emb_model = MLPEmbMapping(layers)

    elif state_emb_model_type == 'Conv':
        assert obs_type == 'cont' and isinstance(obs_dim, tuple) and len(obs_dim) == 3
        layers = [state_emb_hidden_size] * state_emb_layers + [emb_dim]
        state_emb_model = ConvEmbMapping(obs_dim, layers)

    birth_model = GoalDirectedStochPolicy(
        policy_model=MixedEmbMapping(
            emb_model_dict={
                'obs': state_emb_model,
                'emb': IDEmbMapping(emb_dim)},
                emb_dim=act_dim,
                comb_model=MLPEmbMapping([emb_dim + state_emb_model.emb_dim,
                                          birth_model_hs,
                                          int(birth_model_hs / 2),
                                          act_dim], batch_norm=False)))

    # Get the population model tying them together
    pop_model = PopModel(emb_space, metric, birth_model).to(device)

    # Declare optimizers
    optims = {}
    if opt_cls == 'Adam':
        optims['emb_model'] = Adam(emb_model.parameters(), lr=isom_lr)
        optims['birth_model'] = Adam(birth_model.parameters(), lr=birth_lr)
    elif opt_cls == 'SGD':
        optims['emb_model'] = SGD(emb_model.parameters(), lr=isom_lr)
        optims['birth_model'] = SGD(birth_model.parameters(), lr=birth_lr)
    else:
        raise ValueError('{} not a supported opt_cls'.format(opt_cls))

    '''
    learn_embedding(pop_model, isom_epochs=isom_epochs, isom_bs=isom_bs,
                    birth_epochs=birth_epochs, birth_bs=birth_bs,
                    optims=optims, save_dir=save_dir)
    '''

    # Get Initial Embs
    pop_embs, pop_distns = pop_model.init_embs(epochs=isom_epochs, bs=isom_bs, optim=optims['emb_model'], save_dir=save_dir)

    # Train Birth Model
    train_birth_model(pop_model.birth_model,
                      obs=unique_obs,
                      embs=pop_embs.to(device),
                      pop_distns=pop_distns.to(device),
                      optim=optims['birth_model'], epochs=birth_epochs, bs= birth_bs,
                      save_dir=save_dir, save_every=save_every, print_every=print_every)

    torch.save(pop_model, os.path.join(save_dir, 'pop_model'))

    pop_model = torch.load(os.path.join(save_dir, 'pop_model'))

    # Make plots of emb space
    pop_model_plots(pop_model, env, save_dir=save_dir)


def pop_model_plots(pop_model, env, save_dir):

    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plots of emb space coloured by score
    emb_space_score_plots(pop_model, env, save_dir=os.path.join(save_dir, 'score_plots'))

def emb_space_score_plots(pop_model, env, save_dir):
    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from isometric_ea import eval_pop
    import numpy as np
    pop_embs = pop_model.emb_model.model.weight.data

    score_path = os.path.join(save_dir, 'pop_scores')
    if os.path.exists(score_path):
        pop_scores = torch.load(os.path.join(save_dir, 'pop_scores'))
        assert len(pop_scores) == len(pop_embs)
    else:
        print('Computing Scores')
        pop_scores = eval_pop(pop_embs, pop_model=pop_model, env=env, max_ep_len=30, n_reps=30, track=True)
        torch.save(pop_scores, score_path)

    pop_embs, pop_scores = pop_embs.cpu().numpy(), pop_scores.cpu().numpy()

    max_ind = pop_scores.argmax()
    best_emb = pop_model.pop_embs[max_ind].unsqueeze(0).cpu().numpy()

    from matplotlib import pyplot as plt
    print('Generating Emb Score Plots')
    for i in tqdm(range(pop_model.emb_dim)):
        for j in range(i+1, pop_model.emb_dim):
            plt.scatter(pop_embs[:,i], pop_embs[:, j], s=1, c=pop_scores, cmap='hot_r')
            plt.scatter(best_emb[:,i], best_emb[:,j], c='g')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Embedding Dimension {}'.format(i))
            plt.ylabel('Embedding Dimension {}'.format(j))
            plt.title('Population Embeddings Coloured by Fitness')
            #plt.legend(loc='lower right')
            plt.savefig(os.path.join(save_dir, '{}_{}'.format(i,j)))
            plt.close()

    # Projecting onto direction of steepest ascent
    from models.embedding.utils import get_gradient_steepest_ascent
    d1vec, residuals, _ = get_gradient_steepest_ascent(embs=pop_embs, signal=pop_scores)
    d1 = pop_embs.dot(d1vec)

    d2vec, _, _ = get_gradient_steepest_ascent(embs=pop_embs, signal=residuals)
    d2 = pop_embs.dot(d2vec)

    best_emb_proj_1, best_emb_proj_2 = np.dot(best_emb, d1vec), np.dot(best_emb, d2vec)

    plt.scatter(d1, d2, c=pop_scores, cmap='hot_r', s=1)
    title = 'Embs Realigned by {}'.format(' Fitness Score')
    #plt.scatter(best_emb_proj_1, best_emb_proj_2, c='g')
    plt.xlabel('Axis of Steepest Ascent')
    plt.ylabel('Residual Axis of Steepest Ascent')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'embs_score_projection'))
    plt.close()

    # Histogram
    n_bins = 20
    bins = d1.min() + (d1.max() - d1.min()) / n_bins * torch.arange(n_bins + 1).float()
    hist = torch.zeros(n_bins)
    for i in range(n_bins):
        b_mask = (bins[i] <= torch.tensor(d1)) * (torch.tensor(d1) < bins[i + 1])
        hist[i] = torch.tensor(pop_scores).masked_select(b_mask).mean()

    plt.bar(x=bins[:-1].cpu().numpy(), height=hist.cpu().numpy(), align='edge', width=(bins[1]-bins[0]).cpu().numpy())
    title = 'Histogram of {} Along Aligned Axis'.format('Fitness Scores')
    plt.xlabel('Axis of Steepest Ascent')
    plt.ylabel('Average {}'.format('Fitness Score'))
    plt.title(title)
    plt.xticks([])
    plt.savefig(os.path.join(save_dir, 'embs_score_histogram'))
    plt.close()

    plt.scatter(d1, pop_scores)
    plt.title('Fitness Score vs. Axis of Steepest Ascent')
    plt.ylabel('Fitness Score')
    plt.xlabel('Axis of Steepest Ascent')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, 'scores_best_dim'))
    plt.close()


'''
def learn_embedding(pop_model, optims, isom_epochs=10000, isom_bs=100,
                    birth_epochs=100, birth_bs=32, save_dir=None):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert isinstance(optims, dict) and 'emb_model' in optims.keys()

    # Get initial embs
    pop_embs, pop_distns = pop_model.init_embs(epochs=isom_epochs, bs=isom_bs, optim=optims['emb_model'], save_dir=save_dir)

    # Train the birth model in initial distributions
    assert 'birth_model' in optims.keys()
    train_birth_model(pop_model.birth_model, pop_embs, pop_distns, optim=optims['birth_model'], epochs=birth_epochs, bs= birth_bs)

def init_model(pop_model, pop_size, init_sample_size=10000):

    # Generate first population

    # Learn isometric embedding of initial sample
    pop_embs = pop_model.init_pop_model(init_sample_size)

    assert isinstance(pop_size, int) and pop_size > 0
    
    #pop_embs = MultivariateNormal(loc=torch.zeros(emb_dim), covariance_matrix=torch.eye(emb_dim)).sample([pop_size])

    # Learn birth model 

    return pop_embs

def next_generation(pop_model, parent_pop, scores, pop_size=100):

    assert issubclass(type(parent_pop), torch.Tensor) and issubclass(type(scores), torch.Tensor)
    assert scores.dim() == 1 and len(scores) == len(parent_pop)


def learn_pop_embs(emb_model):
    return

def eval_pop(pop_embs, **kwargs):
    
    assert issubclass(type(pop_embs), torch.Tensor)
    scores = torch.zeros(len(pop_embs))
    
    for i in range(len(pop_embs)):
        scores[i] += score_fitness(pop_embs[i], **kwargs)

    return scores

def score_fitness(agent, env, max_ep_len=30, n_reps=50, **kwargs):

    ep_scores = torch.zeros(n_reps)
    for rep in range(n_reps):
        ep_reward, _ = play_episode(agent, env, max_ep_len=max_ep_len)
        ep_scores[rep] += ep_reward

    return ep_scores.mean()

def play_episode(env, agent, max_ep_len, step_buffer=None, **kwargs):

    assert issubclass(type(env), GoalMDPWrapper) and env.n_tasks == 1
    obs, task_id = env.reset()
    task_goal_obs = env.goal_states[task_id]

    # Need to decide what to do about worker randomness
    epsilon = kwargs.get('epsilon', DEFAULT_EPSILON)

    #import pdb; pdb.set_trace()

    ep_len = 0
    done = False
    ep_reward = 0
    obs_list, act_list = [], []
    while ep_len < max_ep_len and not done:
        # predict action
        with torch.no_grad():
            act = agent.get_action(obs=obs.unsqueeze(0).to(device),
                                   epsilon=epsilon)

        # take action
        next_obs, reward, done, info = env.step(act)

        obs_list.append(obs)
        act_list.append(act)

        # record step info
        step = {
            'obs': obs,
            'act': act,
            'reward': torch.tensor(reward),
            'next_obs': next_obs,
            'done': done}
        if step_buffer is not None:
            step_buffer.add_item(deepcopy(step))

        obs = deepcopy(next_obs)
        ep_len += 1
        ep_reward += reward

    obs_list.append(obs)
    return ep_reward, {'obs_list': obs_list,
                       'act_list': act_list,
                       'goal_obs': task_goal_obs}


def plot_embs(emb_model, obs, save_dir=None, save_name=None, batch_size=100, n_dims=2,
              comp_scores=None, rewards=None, successes=None, plt_trajectory=False,
              env=None, agent=None):

    import numpy as np
    from matplotlib import pyplot as plt
    hist_len = 100

    #import pdb; pdb.set_trace()

    assert save_dir is not None and save_name is not None
    eigval_save_dir = os.path.join(save_dir, 'eigenvalues')
    emb_save_dir = os.path.join(save_dir, 'SVD_embs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(eigval_save_dir):
        os.makedirs(eigval_save_dir)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    if plt_trajectory:
        traj_save_dir = os.path.join(save_dir, save_name)
        if not os.path.exists(traj_save_dir):
            os.makedirs(traj_save_dir)

    n_samples = obs.size(0)
    emb_dim = emb_model.emb_dim
    embs = torch.zeros((n_samples, emb_dim)).cuda()

    assert isinstance(batch_size, int) and batch_size > 0
    for i in range(0, n_samples, batch_size):
        with torch.no_grad():
            batch_embs = emb_model(obs[i:(i + batch_size)])
        embs[i:(i + batch_size), :] = batch_embs

    embs = embs.cpu().numpy()
    emb_mean = np.mean(embs, 0)
    embs -= emb_mean

    embs_pca, eigvals, eigvecs = pca(embs)
    #embs_pca = embs # must change
    
    #U, S, V = np.linalg.svd(embs, full_matrices=False, compute_uv=True)
    #embs_svd = np.dot(U, np.diag(S))    
    #eigvals = S * S / (n_samples - 1)
    
    plt.bar([i for i in range(emb_dim)], eigvals)
    plt.savefig(os.path.join(eigval_save_dir, save_name))
    plt.close()

    for i in range(n_dims):
        for j in range(i+1, n_dims):
            plt.scatter(embs_pca[:,i], embs_pca[:, j], s=1)
            plt.savefig(os.path.join(emb_save_dir, save_name + '_{}_{}'.format(i,j)))
            plt.close()

    if plt_trajectory:
        assert env is not None and agent is not None
        _, (obs_list, act_list, goal_obs) = play_episode(env, emb_model, agent, step_buffer=None, max_ep_len=50, epsilon=0.)
        save_trajectory(emb_model, obs_list, act_list, goal_obs,
                        embs_pca=embs_pca, emb_mean=emb_mean, eigvecs=eigvecs, save_dir=traj_save_dir)

    if comp_scores is not None and len(comp_scores) > hist_len:
        assert isinstance(comp_scores, np.ndarray)
        smooth_scores = np.convolve(comp_scores, np.ones((hist_len,))/hist_len, mode='valid')
        plt.plot([i for i in range(len(smooth_scores))], smooth_scores)
        plt.title('Completion Scores')
        plt.savefig(os.path.join(save_dir, 'comp_scores'))
        plt.close()

    if rewards is not None and len(rewards) > hist_len:
        assert isinstance(rewards, np.ndarray)
        smooth_rews = np.convolve(rewards, np.ones((hist_len,))/hist_len, mode='valid')
        plt.plot([i for i in range(len(smooth_rews))], smooth_rews)
        plt.title('Rewards')
        plt.savefig(os.path.join(save_dir, 'rewards'))
        plt.close()

    if successes is not None and len(successes) > hist_len:
        assert isinstance(successes, np.ndarray)
        smooth_succs = np.convolve(successes, np.ones((hist_len,))/hist_len, mode='valid')
        plt.plot([i for i in range(len(smooth_succs))], smooth_succs)
        plt.title('Success Rate')
        plt.savefig(os.path.join(save_dir, 'success_rate'))
        plt.close()


def save_trajectory(emb_model, obs_list, act_list, goal_obs, embs_pca, emb_mean, eigvecs, save_dir, **kwargs):

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    with torch.no_grad():
        traj_embs = [emb_model(obs.to(device).unsqueeze(0)).cpu().numpy() - emb_mean for obs in obs_list]
        traj_embs_pca = [traj_emb.dot(eigvecs) for traj_emb in traj_embs]
        goal_pca = (emb_model(goal_obs.to(device).unsqueeze(0)).cpu().numpy() - emb_mean).dot(eigvecs)

    traj_embs_pca = np.concatenate(traj_embs_pca)

    # Save first obs and goal obs
    plt.imshow(obs_list[0].cpu().numpy().transpose([1,2,0]))
    plt.savefig(os.path.join(save_dir, 'start_obs'))
    plt.close()
    plt.imshow(goal_obs.cpu().numpy().transpose([1,2,0]))
    plt.savefig(os.path.join(save_dir, 'goal_obs'))
    plt.close()

    # plotting all together
    for i in range(traj_embs_pca.shape[0]):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        x, y = list(traj_embs_pca[:,0]), list(traj_embs_pca[:, 1])
        line = Line2D(x, y)
        ax2.add_line(line)
        ax2.set_xlim(min(list(embs_pca[:,0]))-1, max(list(embs_pca[:,0]))+1)
        ax2.set_ylim(min(list(embs_pca[:,1]))-1, max(list(embs_pca[:,1]))+1)
        ax2.scatter(embs_pca[:,0], embs_pca[:,1],s=1)
        ax2.scatter(traj_embs_pca[i,0], traj_embs_pca[i,1],c='red')
        ax2.scatter(goal_pca[:,0], goal_pca[:,1],c='green')
        #ax2.set_title('Trajectory Path Dims {}-{}'.format(0, 1))
        ax2.set_xlabel('PCA Dim 0')
        ax2.set_ylabel('PCA Dim 1')
        #ax2.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off')
        ax2.set_xticks([])
        ax2.set_yticks([])
        x, y = list(traj_embs_pca[:,0]), list(traj_embs_pca[:, 2])
        line = Line2D(x, y)
        ax3.add_line(line)
        ax3.set_xlim(min(list(embs_pca[:,0]))-1, max(list(embs_pca[:,0]))+1)
        ax3.set_ylim(min(list(embs_pca[:,2]))-1, max(list(embs_pca[:,2]))+1)
        ax3.scatter(embs_pca[:,0], embs_pca[:,2],s=1)
        ax3.scatter(traj_embs_pca[i,0], traj_embs_pca[i,2],c='red')
        ax3.scatter(goal_pca[:,0], goal_pca[:,2],c='green')
        #ax3.set_title('Trajectory Path Dims {}-{}'.format(0, 2))
        ax3.set_xlabel('PCA Dim 0')
        ax3.set_ylabel('PCA Dim 2')
        #ax3.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off')
        ax3.set_xticks([])
        ax3.set_yticks([])
        x, y = list(traj_embs_pca[:,1]), list(traj_embs_pca[:, 2])
        line = Line2D(x, y)
        ax4.add_line(line)
        ax4.set_xlim(min(list(embs_pca[:,1]))-1, max(list(embs_pca[:,1]))+1)
        ax4.set_ylim(min(list(embs_pca[:,2]))-1, max(list(embs_pca[:,2]))+1)
        ax4.scatter(embs_pca[:,1], embs_pca[:,2],s=1)
        ax4.scatter(traj_embs_pca[i,1], traj_embs_pca[i,2],c='red')
        ax4.scatter(goal_pca[:,1], goal_pca[:,2],c='green')
        #ax4.set_title('Trajectory Path Dims {}-{}'.format(1, 2))
        ax4.set_xlabel('PCA Dim 1')
        ax4.set_ylabel('PCA Dim 2')
        #ax4.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax1.imshow(obs_list[i].cpu().numpy().transpose([1,2,0]))
        ax1.set_xlabel('MDP State')
        #ax1.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.savefig(os.path.join(save_dir, 'pca_emb_trajectory_states_{}'.format(i)))
        plt.close()
'''

def save_trajectory(emb_model, obs_list, act_list, goal_obs, embs_pca, emb_mean, eigvecs, save_dir, **kwargs):

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    with torch.no_grad():
        traj_embs = [(emb_model(obs.to(device).unsqueeze(0)).cpu().numpy() - emb_mean).dot(eigvecs) for obs in obs_list]
        goal_emb = (emb_model(goal_obs.to(device).unsqueeze(0)).cpu().numpy() - emb_mean).dot(eigvecs)

    traj_embs = np.concatenate(traj_embs)

    # plotting all together
    for i in range(traj_embs.shape[0]):
        fig, ax1 = plt.subplots(1)
        x, y = list(traj_embs[:,0]), list(traj_embs[:, 1])
        line = Line2D(x, y)
        ax1.add_line(line)
        ax1.set_xlim(min(list(embs_pca[:,0])), max(list(embs_pca[:,0])))
        ax1.set_ylim(min(list(embs_pca[:,1])), max(list(embs_pca[:,1])))
        ax1.scatter(embs_pca[:,0], embs_pca[:,1],s=1)
        ax1.scatter(traj_embs[i,0], traj_embs[i,1],c='red')
        ax1.scatter(goal_emb[:,0], goal_emb[:,1],c='green')
        #ax2.set_title('Trajectory Path Dims {}-{}'.format(0, 1))
        ax1.set_xlabel('Emb Dim 0')
        ax1.set_ylabel('Emb Dim 1')
        #ax2.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.savefig(os.path.join(save_dir, 'pca_emb_trajectory_states_{}'.format(i)))
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data Arguments
    parser.add_argument('--env_name', type=str, default=None, choices=ENVS)

    # Embedding Space Arguments
    parser.add_argument('--emb_space_type',
                        help='what type of embedding space to use',
                        type=str, default='euclidean',
                        choices=EMBEDDING_SPACES)
    parser.add_argument('--metric_type',
                        help='what type of metric to use on real data',
                        type=str, default='TV',
                        choices=['TV', 'JSD'])

    # Birth Model Arguments
    parser.add_argument('--state_emb_model_type',
                        help='what type of emb model to use',
                        type=str, default='MLP',
                        choices=['MLP', 'Discrete', 'Conv'])
    parser.add_argument('--state_emb_hidden_size', help='dimension of hidden layer of state embedding',
                        type=int, default=12)
    parser.add_argument('--state_emb_layers', help='number of hidden layers for state embedding',
                        type=int, default=1)
    parser.add_argument('--birth_model_hs', help='dimension of hidden layer of policy',
                        type=int, default=2000)

    # Emb Model Specifications
    parser.add_argument('--emb_dim', help='size of embeddings',
                        type=int, default=12)
    
    
    # Optimization Arguments
    parser.add_argument('--pop_size', help='buffer size', type=int, default=2500)
    parser.add_argument('--opt_cls', help='which optimizer type to use',
                        type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--isom_bs', help='batch_size for isometric embedding', type=int, default=100)
    parser.add_argument('--isom_epochs', help='number of epochs for isometric embedding', type=int, default=1000)
    parser.add_argument('--isom_lr', help='isometric embedding learning rate', type=float, default=0.01)
    parser.add_argument('--birth_lr', help='isometric embedding learning rate', type=float, default=0.001)
    parser.add_argument('--birth_epochs', help='number of epochs for birth model', type=int, default=25)
    parser.add_argument('--birth_bs', help='batch_size for birth_model', type=int, default=32)

    # Recording Args
    parser.add_argument('--model_dir', help='directory to save models in', type=str, default=None)
    parser.add_argument('--name', help='what to name model', type=str, default=None)
    parser.add_argument('--save_every', help='how often to save (epochs)', type=int, default=5)
    parser.add_argument('--print_every', help='how often to print (epochs)', type=int, default=1)
    parser.add_argument('--seed', help='random seed', type=str, default=None)
    parser.add_argument('--plt', dest='plt', default=False, action='store_const', const=True)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    main(**args)
