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

import multiprocessing as mp

import torch
#from torch.optim import Adam, SGD
#from torch.utils.data import DataLoader, TensorDataset
#from torch.distributions import MultivariateNormal
from torch.nn.functional import softmax
from torch.distributions import Categorical, MultivariateNormal

from models.rl.envs import ENVS, get_mdp
from models.pop_model import PopModel, GoalPolicyWrapper
from models.agent import GoalDirectedStochPolicy
from models.rl.goal_directed_rl import GoalMDPWrapper, get_goal_mdp_wrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(env_name, pop_model_name=None,
         pop_size=100, n_generations=5000, max_ep_len=100, opt_cls='Adam',
         unique_samples=2000, isom_epochs=10000, isom_bs=32, isom_lr=0.001,
         model_dir=None, name=None, save_every=5, print_every=1, seed=None, plt=False):

    if seed is not None:
        torch.manual_seed(seed)

    if model_dir is None:
        model_dir = os.path.join(os.getcwd(), 'trained_models')

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    assert pop_model_name is not None and isinstance(pop_model_name, str)
    pop_model_dir = os.path.join(model_dir, env_name, pop_model_name)
    assert os.path.exists(pop_model_dir)

    assert name is not None and isinstance(name, str)
    assert env_name in ENVS

    save_dir = os.path.join(model_dir, env_name, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all arguments of the current training run for inspection later if desired
    logging.basicConfig(filename=os.path.join(save_dir, 'train.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
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
    #if not os.path.exists(env_path):
        #env, obs_type, obs_dim, act_dim = get_goal_mdp_wrapper(env_name, n_tasks=1)
        #env.save(env_path)
    #else:
        #_, obs_type, obs_dim, act_dim = get_goal_mdp_wrapper(env_name, n_tasks=1)
        #env = GoalMDPWrapper.load(env_path)

    # Get the population model
    pop_model = torch.load(os.path.join(pop_model_dir, 'pop_model')).to(device)
    torch.save(pop_model, os.path.join(save_dir, 'pop_model'))

    '''
    # Declare optimizer
    optims = {}
    if opt_cls == 'Adam':
        optims['policy'] = Adam(policy.parameters(), lr=policy_lr        optims['value'] = Adam(value_fn.parameters(), lr=value_lr)
    elif opt_cls == 'SGD':
        optims['policy'] = SGD(policy.parameters(), lr=policy_lr)
        optims['value'] = SGD(value_fn.parameters(), lr=value_lr)
    else:
        raise ValueError('{} not a supported opt_cls'.format(opt_cls))
    '''

    # Train Models
    #agent = torch.load('trained_models/fourrooms/alt_run11/agent').to(device)
    #emb_model = torch.load('trained_models/fourrooms/alt_run11/emb_model').to(device)

    run_ea(n_generations, pop_size, pop_model, env, save_dir=save_dir)

def run_ea(n_generations, pop_size, pop_model, env, save_dir=None, **kwargs):
    '''
    Learning routine:
        1. Evaluate population fitness scores
        2. Select candidates for reproduction and generate offspring
    '''
    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set up recording
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    generation_scores = torch.zeros(n_generations)
    all_embs = torch.zeros((n_generations + 1, pop_size, pop_model.emb_dim))
    all_scores = torch.zeros((n_generations + 1, pop_size))
    all_embs_flat = torch.zeros(((n_generations + 1) * pop_size, pop_model.emb_dim))
    all_scores_flat = torch.zeros(((n_generations + 1) * pop_size))
    momentums = torch.zeros(n_generations, pop_model.emb_dim)

    # Initialize population of embeddings
    pop_embs = init_population(pop_model, pop_size)

    for g in tqdm(range(n_generations)):
        # Learn embedding and reconstruction model
        # pop_model.learn_pop_embs(pop_embs)

        # Evaluate fitenss scores
        scores = eval_pop(pop_embs, pop_model=pop_model, env=env, **kwargs)

        logger.info('Generation {} Mean Score: {}'.format(g, scores.mean()))
        generation_scores[g] += scores.mean()
        plot_scores(generation_scores[:g+1].cpu().numpy(), save_dir=save_dir, save_name=g)

        # Record pop embs and scores for graphing
        all_scores[g] += scores.clone().cpu()
        all_embs[g] += pop_embs.clone().cpu()
        all_embs_flat[g * pop_size:(g+1) * pop_size] += pop_embs.clone().cpu()
        all_scores_flat[g * pop_size:(g+1) * pop_size] += scores.clone().cpu()

        # Get direction of steepest ascent
        momentum_vec = get_momentum_vector(all_embs_flat[:(g+1) * pop_size], all_scores_flat[:(g+1) * pop_size])
        momentums[g] += momentum_vec.clone()

        # Generate new population
        pop_embs = new_generation_ascent(parent_pop_embs=pop_embs, scores=scores, pop_size=pop_size, momentum=momentum_vec)

    scores = eval_pop(pop_embs, pop_model=pop_model, env=env, **kwargs)

    all_scores[-1] += scores.clone().cpu()
    all_embs_flat[-pop_size:] += pop_embs.clone().cpu()
    all_scores_flat[-pop_size:] += scores.clone().cpu()

    torch.save(generation_scores, os.path.join(save_dir, 'generation_scores'))
    torch.save(all_scores, os.path.join(save_dir, 'all_scores'))
    torch.save(all_embs, os.path.join(save_dir, 'all_embs'))
    torch.save(all_embs_flat, os.path.join(save_dir, 'all_embs_flat'))
    torch.save(all_scores_flat, os.path.join(save_dir, 'all_scores_flat'))
    torch.save(momentums, os.path.join(save_dir, 'momentums'))
    
    plot_progression(all_embs.numpy(), all_scores.numpy(), momentums=momentums.cpu().numpy(),
                     save_dir=os.path.join(save_dir, 'training_progression'))


def init_population(pop_model, pop_size):

    all_embs = pop_model.emb_model.model.weight.data
    total_pop_size = all_embs.size(0)
    assert total_pop_size >= pop_size
    rand_inds = torch.randint(0, total_pop_size,size=(pop_size,))

    return all_embs[rand_inds]

def new_generation_ascent(parent_pop_embs, scores, pop_size=100, momentum=None):

    assert issubclass(type(parent_pop_embs), torch.Tensor) and issubclass(type(scores), torch.Tensor)
    assert scores.dim() == 1 and len(scores) == len(parent_pop_embs)

    child_pop_embs = select_children(parent_pop_embs, scores, pop_size)

    return mutate_ascent(child_pop_embs, scores, momentum=momentum)

def eval_pop(pop_embs, pop_model, env, **kwargs):
    
    assert issubclass(type(pop_embs), torch.Tensor)
    scores = torch.zeros(len(pop_embs))

    #q = mp.Queue()
    #arg_iter = []
    #for i in range(len(pop_embs)):
        #arg_iter.append({'emb': pop_embs[i], 'pop_model': pop_model, 'env': env})
        #arg_iter.append((pop_embs[i], pop_model, env))
        #q.put({'emb': pop_embs[i], 'pop_model': pop_model, 'env': env})

    #p = mp.Process(target=score_fitness, kwargs=q)
    #p.start()
    #print(q.get())
    #p.join()
    #int(0.66 * mp.cpu_count())
    #import pdb; pdb.set_trace()
    #with mp.Pool(processes=20) as pool:
    #    scores = pool.starmap_async(score_fitness, arg_iter).get()
    for i in range(len(pop_embs)):
        scores[i] += score_fitness(emb=pop_embs[i], pop_model=pop_model, env=env, **kwargs)

    return scores

def score_fitness(emb, pop_model, env, max_ep_len=30, n_reps=50, **kwargs):

    agent = GoalPolicyWrapper(goal_policy=pop_model.birth_model, emb=emb)
    ep_scores = torch.zeros(n_reps)
    for rep in range(n_reps):
        ep_reward, _ = play_episode(env, agent, max_ep_len=max_ep_len)
        ep_scores[rep] += ep_reward

    return ep_scores.mean()

def play_episode(env, agent, max_ep_len, step_buffer=None, **kwargs):

    #assert issubclass(type(env), GoalMDPWrapper) and env.n_tasks == 1
    obs = env.reset()
    #task_goal_obs = env.goal_states[task_id]

    ep_len = 0
    done = False
    ep_reward = 0
    obs_list, act_list = [], []
    while ep_len < max_ep_len and not done:
        # predict action
        with torch.no_grad():
            act = agent.get_action(obs=obs.unsqueeze(0).to(device))

        # take action
        next_obs, reward, done, info = env.step(act)

        '''
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
        '''

        obs = deepcopy(next_obs)
        ep_len += 1
        ep_reward += reward

    #obs_list.append(obs)
    return ep_reward, {}


def select_children(parent_pop_embs, scores, pop_size, temp=1.):
    assert issubclass(type(parent_pop_embs), torch.Tensor) and issubclass(type(scores), torch.Tensor)
    assert isinstance(temp, float) and temp > 0.
    selection_dist = Categorical(probs=softmax(scores / temp, dim=-1))
    selected_inds = selection_dist.sample([pop_size, 2])
    selected_parents = parent_pop_embs[selected_inds]

    return selected_parents.mean(dim=1)

def mutate_ascent(pop_embs, scores, stdev=0.03, step_size=0.01, momentum=None, **kwargs):
    assert issubclass(type(pop_embs), torch.Tensor)
    assert isinstance(stdev, float) and stdev > 0.

    emb_dim = pop_embs.size(-1)
    if momentum is not None:
        mutation_mean = momentum
    else:
        mutation_mean = torch.zeros(emb_dim)

    #from models.embedding.utils import get_gradient_steepest_ascent
    #ascent_vec, _ = get_gradient_steepest_ascent(embs=pop_embs.cpu().numpy(), signal=scores.cpu().numpy(), norm=True)
    #ascent_vec = torch.from_numpy(ascent_vec)

    # if emb_space is not None:
    #     from itertools import combinations
    #     [I, J] = [list(idx) for idx in zip(*list(combinations(range(len(pop_embs)), 2)))]
    #     max_dist = emb_space.metric(pop_embs[I], pop_embs[J]).max()
    #     mutation_mean = step_size * max_dist * ascent_vec
    # else:
    #     mutation_mean = step_size * ascent_vec


    mutation_dist = MultivariateNormal(loc=mutation_mean.cpu(),
                                       covariance_matrix=(stdev ** 2) * torch.eye(emb_dim))

    return pop_embs + mutation_dist.sample([pop_embs.size(0)]).to(pop_embs.device.type)

def get_momentum_vector(pop_embs, scores, conf_thresh=0.1, step_size=0.03):
    from models.embedding.utils import get_gradient_steepest_ascent
    ascent_vec, _, p_val = get_gradient_steepest_ascent(embs=pop_embs.cpu().numpy(),
                                                       signal=scores.cpu().numpy(), norm=True)
    if p_val < conf_thresh:
        return torch.from_numpy(step_size * ascent_vec)
    else:
        return torch.zeros(ascent_vec.shape[0])


def plot_scores(scores=None, save_dir=None, save_name=None, batch_size=100, n_dims=2,
                plt_trajectory=False, env=None, agent=None):

    import numpy as np
    from matplotlib import pyplot as plt
    hist_len = 1

    #import pdb; pdb.set_trace()

    assert save_dir is not None and save_name is not None
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if plt_trajectory:
        traj_save_dir = os.path.join(save_dir, save_name)
        if not os.path.exists(traj_save_dir):
            os.makedirs(traj_save_dir)

    if plt_trajectory:
        assert env is not None and agent is not None
        _, (obs_list, act_list, goal_obs) = play_episode(env, emb_model, agent, step_buffer=None, max_ep_len=50, epsilon=0.)
        save_trajectory(emb_model, obs_list, act_list, goal_obs,
                        embs_pca=embs_pca, emb_mean=emb_mean, eigvecs=eigvecs, save_dir=traj_save_dir)

    if scores is not None and len(scores) > hist_len:
        assert isinstance(scores, np.ndarray)
        smooth_scores = np.convolve(scores, np.ones((hist_len,))/hist_len, mode='valid')
        plt.plot([i for i in range(len(smooth_scores))], smooth_scores)
        plt.title('Generation Scores')
        plt.savefig(os.path.join(save_dir, 'scores'))
        plt.close()

def plot_progression(all_embs, all_scores, momentums, save_dir=None):
    
    assert save_dir is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Plotting Training Progression')
    from matplotlib import pyplot as plt
    from models.embedding.utils import pca, images_to_gif, get_gradient_steepest_ascent
    from itertools import combinations
    import numpy as np

    n_gens, pop_size, emb_dim = all_embs.shape[0], all_embs.shape[1], all_embs.shape[2]
    all_embs_flat = np.zeros((n_gens * pop_size, emb_dim))
    all_scores_flat = np.zeros(n_gens * pop_size)
    for i in range(n_gens):
        for j in range(pop_size):
            all_embs_flat[i*pop_size+j] += all_embs[i, j]
            all_scores_flat[i*pop_size+j] += all_scores[i, j]

    emb_mean = np.mean(all_embs_flat, axis=0)
    embs_pca, _, eig_vecs = pca(all_embs_flat - emb_mean)

    avg_scores = all_scores.mean(axis=1)

    x_lim = (embs_pca[:, 0].min() - 0.05 * (embs_pca[:, 0].max() - embs_pca[:, 0].min()), embs_pca[:, 0].max() + 0.05 * (embs_pca[:, 0].max() - embs_pca[:, 0].min()))
    y_lim = (embs_pca[:, 1].min() - 0.05 * (embs_pca[:, 1].max() - embs_pca[:, 1].min()), embs_pca[:, 1].max() + 0.05 * (embs_pca[:, 1].max() - embs_pca[:, 1].min()))
    file_names = []
    for g in tqdm(range(n_gens)):
        # Plot all old embs first
        plt.clf()
        #fig, (ax1, ax2) = plt.subplots(1,2)
        fig, ax1 = plt.subplots(1)
        ax1.scatter(embs_pca[:g*pop_size,0], embs_pca[:g*pop_size, 1], s=1, c=all_scores_flat[:g*pop_size], cmap='hot_r')

        # Plot current generation
        ax1.scatter(embs_pca[g*pop_size:(g+1)*pop_size,0], embs_pca[g*pop_size:(g+1)*pop_size, 1],
                    s=1, c=all_scores_flat[g*pop_size:(g+1)*pop_size], cmap='hot_r', edgecolors='black')

        #import pdb; pdb.set_trace()
        # trail_len = 0
        # min_ind = 0 #if g < trail_len else g - trail_len
        # [I, J] = [list(idx) for idx in zip(*list(combinations(range(pop_size), 2)))]
        # max_dist = 1 #np.linalg.norm((all_embs_flat[min_ind*pop_size:(g+1)*pop_size][I], all_embs_flat[min_ind*pop_size:(g+1)*pop_size][J])).max()
        # step_size = 0.1 * max_dist
        # ascent_vec, _ = get_gradient_steepest_ascent(embs=all_embs_flat[min_ind*pop_size:(g+1)*pop_size], signal=all_scores_flat[min_ind*pop_size:(g+1)*pop_size], norm=True)
        # ascent_vec = np.dot(step_size * ascent_vec, eig_vecs)

        arrow_start = embs_pca[g*pop_size:(g+1)*pop_size].mean(axis=0)
        if g < n_gens - 1:
            momentum_pca = (momentums[g] - emb_mean).dot(eig_vecs)
        else:
            momentum_pca = np.zeros(2)

        dx, dy = momentum_pca[0], momentum_pca[1]
        ax1.arrow(x=arrow_start[0], y=arrow_start[1], dx=dx, dy=dy)

        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)

        ax1.set_xticks([])
        ax1.set_yticks([])

        # ax2.plot([i for i in range(g + 1)], avg_scores[:(g+1)], c='r')
        # # ['b'] * (g + 1) + ['w'] * (n_gens - g - 1)
        # ax2.set_xlim((0, n_gens))
        # ax2.set_ylim((-0.1, 1.1))
        # ax2.set_xticks([])
        # ax2.set_yticks([])

        fig.suptitle('Training Progression')
        fig.text(0.5, 0.01, 'Generation {}'.format(g), va='bottom', ha='center')
        fig.set_figwidth(8)
        fig.set_figheight(4)
        
        plt.savefig(os.path.join(save_dir, 'gen_{}'.format(g)))
        file_names.append('gen_{}.png'.format(g))
        plt.close()

    images_to_gif(img_dir=save_dir, file_names=file_names, out_name='training', fps=10)


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
    parser.add_argument('--pop_model_name', help='which agent embedding to load', type=str, default=None)
    
    # Optimization Arguments
    parser.add_argument('--max_ep_len', help='cutoff point for episode', type=int, default=100)
    parser.add_argument('--n_generations', help='buffer size', type=int, default=1000)
    parser.add_argument('--pop_size', help='buffer size', type=int, default=100)
    parser.add_argument('--opt_cls', help='which optimizer type to use',
                        type=str, default='Adam', choices=['Adam', 'SGD'])

    '''
    # Isomap arguments
    
    parser.add_argument('--isomap_samples', help='number of state samples for isomap', type=int, default=1000)
    parser.add_argument('--isom_bs', help='batch_size for isometric embedding', type=int, default=32)
    parser.add_argument('--isom_epochs', help='number of epochs for isometric embedding', type=int, default=10000)
    parser.add_argument('--isom_lr', help='isometric embedding learning rate', type=float, default=0.001)
    '''
    

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
    mp.set_start_method('spawn')
    args = parse_args()
    main(**args)
