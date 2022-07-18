import os
import sys
from datetime import date
import time
import json

from multiprocessing import cpu_count

from tqdm import tqdm
import argparse

import itertools
import jax.numpy as jnp
from jax import vmap
from jax.config import config

from joblib import Parallel, delayed

from src.utils import get_state
from src.loader import get_agent_by_name, get_env_by_name, get_kernel_by_name

config.update("jax_enable_x64", True)
today = date.today()

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


def save_result(settings, metrics):
    # results_dir = 'results/{}/{}/{}'.format(settings['env'], settings['exp_name'], today.strftime("%d-%m-%Y"))
    results_dir = 'results/{}'.format(settings['exp_name'])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_name = os.path.join(results_dir, 'metrics' + '_' + str(settings['random_seed_agent']) + '_' +
                             str(settings['random_seed_env']) + '_' + settings['exp_name'] +
                             '.json')
    merged_dict = {**settings, **metrics}
    with open(file_name, 'w') as file:
        file.write(json.dumps({"results": merged_dict}))


def do_single_experiment(parameters, rd_agent, rd_env):
    parameters['random_seed_agent'] = rd_agent
    parameters['random_seed_env'] = rd_env
    print('Running single experiment')
    print("Random seed Agent: {}".format(rd_agent))
    print("Random seed Env: {}".format(rd_env))

    # Instantiation
    kernel_env = get_kernel_by_name(parameters, 'env')
    env = get_env_by_name(parameters, kernel_env)
    kernel_agent = get_kernel_by_name(parameters, 'agent')
    agent = get_agent_by_name(parameters)(parameters, kernel_agent)
    metrics = {'step': [],
               'time': [],
               'average_reward': [],
               'average_reward_clean': [],
               'average_best': [],
               'sum_reward': [],
               'sum_reward_clean': [],
               'sum_best': [],
               'regret': [],
               'regret_noiseless': []
               }
    save_parameters = {k: parameters[k] for k in ['agent',  'kernel_agent', 'kernel_agent_param', 'reg_lambda', 'mu',
                                                  'env', 'kernel_env', 'kernel_env_param',
                                                  'parallelization', 'T',
                                                  'min_action', 'max_action', 'n_actions', 'dim_actions',
                                                  'min_context', 'max_context', 'n_contexts', 'dim_contexts',
                                                  'discrete_contexts',
                                                  'noise_scale',
                                                  'exp_name',
                                                  'random_seed_agent', 'random_seed_env']}
    best_strategy_rewards = []

    # t0
    t0 = time.time()

    # Iterations
    for step in tqdm(range(parameters['T'])):
        # Choose a random context
        context = env.sample_data()
        states_grid = vmap(lambda x: get_state(context, x))(jnp.array(parameters['actions_grid']))
        # Iteration of the agent
        action = agent.sample_action(states_grid)
        state = get_state(context, action)
        reward_clean = env.sample_reward(state)
        reward = env.sample_reward_noisy(reward_clean)
        agent.update_agent(state, reward_clean, reward)
        # Best reward possible
        best_strategy_rewards.append(env.get_best_reward_in_context(context, states_grid).squeeze())

        # Metrics
        if step % 100 == 0 and step != 0:
            metrics['step'].append(step)
            t = time.time() - t0
            metrics['time'].append(t)
            average_reward = jnp.mean(jnp.array(agent.rewards)).item()
            metrics['average_reward'].append(average_reward)
            average_reward_clean = jnp.mean(jnp.array(agent.rewards_clean)).item()
            metrics['average_reward_clean'].append(average_reward_clean)
            average_best = jnp.mean(jnp.array(best_strategy_rewards)).item()
            metrics['average_best'].append(average_best)
            sum_agent = jnp.sum(jnp.array(agent.rewards)).item()
            metrics['sum_reward'].append(sum_agent)
            sum_agent_clean = jnp.sum(jnp.array(agent.rewards_clean)).item()
            metrics['sum_reward_clean'].append(sum_agent_clean)
            sum_best = jnp.sum(jnp.array(best_strategy_rewards)).item()
            metrics['sum_best'].append(sum_best)
            regret = sum_best - sum_agent
            regret_noiseless = sum_best - sum_agent_clean
            metrics['regret'].append(regret)
            metrics['regret_noiseless'].append(regret_noiseless)
            print("Random seed Agent: {}".format(rd_agent))
            print("Random seed Env: {}".format(rd_env))
            print('Step: {}'.format(step))
            print('Average reward: {}'.format(average_reward))
            print('Regret: {}'.format(regret))
            print('Regret Noiseless: {}'.format(regret_noiseless))
            save_result(save_parameters, metrics)


def experiment(args):
    # Setup parameters
    parameters = {
        # Algorithm
        'agent': args.algo,
        'kernel_agent': args.kernel_agent,
        'kernel_agent_param': args.kernel_agent_param,
        'reg_lambda': args.lbd,
        'mu': args.mu,
        # Environment
        'env': args.env,
        'kernel_env': args.kernel_env,
        'kernel_env_param': args.kernel_env_param,
        # Experiment parameters
        'parallelization': args.parallelization,
        'T': args.max_horizon,
        'min_action': args.min_action,
        'max_action': args.max_action,
        'n_actions': args.n_actions,
        'dim_actions': args.dim_actions,
        'min_context': args.min_context,
        'max_context': args.max_context,
        'n_contexts': args.n_contexts,
        'dim_contexts': args.dim_contexts,
        'discrete_contexts': args.discrete_contexts,
        'noise_scale': args.noise_scale,
        'exp_name': args.exp_name
    }
    print(parameters)
    parameters['actions'] = jnp.linspace(args.min_action, args.max_action, args.n_actions).tolist()
    parameters['contexts'] = None
    parameters['actions_grid'] = [[parameters['actions'][i] for i in a] for a in
                                  itertools.product(range(parameters['n_actions']),
                                                    repeat=parameters['dim_actions'])]
    if parameters['discrete_contexts']:
        parameters['contexts'] = jnp.linspace(args.min_context, args.max_context, args.n_contexts).tolist()
    # Parallelization
    if parameters['parallelization']:
        Parallel(n_jobs=cpu_count(), verbose=100)(delayed(do_single_experiment)(parameters, rd_agent, rd_env)
                                                  for (rd_agent, rd_env) in zip(args.rd_seeds_agent, args.rd_seeds_env))
    # Unique CPU
    else:
        for (rd_agent, rd_env) in zip(args.rd_seeds_agent, args.rd_seeds_env):
            do_single_experiment(parameters, rd_agent, rd_env)
    # Merge history
    dict_merge = {"results": []}
    results_dir = 'results/{}'.format(parameters['exp_name'])
    for (rd_agent, rd_env) in zip(args.rd_seeds_agent, args.rd_seeds_env):
        file_name = os.path.join(results_dir, 'metrics' + '_' + str(rd_agent) + '_' +
                                 str(rd_env) + '_' + parameters['exp_name'] +
                                 '.json')
        with open(file_name, 'r') as file:
            dict_merge['results'].append(json.load(file)["results"])
        os.remove(file_name)
    file_name = os.path.join(results_dir, 'metrics' + '_' + parameters['exp_name'] + '_'
                             + today.strftime("%d-%m-%Y") + '.json')
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        with open(file_name, 'r') as file:
            current_info = json.load(file)
            current_info["results"] = current_info["results"] + dict_merge["results"]
        with open(file_name, 'w') as file:
            json.dump(current_info, file)
    else:
        with open(file_name, 'w') as file:
            file.write(json.dumps(dict_merge))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    # Algorithm
    parser.add_argument('--algo', nargs="?", default='k_ucb', choices=['k_ucb', 'random'], help='Algorithm')
    parser.add_argument('--kernel_agent', nargs="?", default='gauss', choices=['gauss', 'exp'],
                        help='Agent Kernel choice')
    parser.add_argument('--kernel_agent_param', nargs="?", default=None)
    parser.add_argument('--lbd', nargs="?", type=float, default=1, help='Regularization parameter')
    parser.add_argument('--mu', nargs="?", type=float, default=1, help='Projection parameter')
    # Environment
    parser.add_argument('--env', nargs="?", default='kernel_linear', choices=['bump', 'kernel_linear'],
                        help='Environment')
    parser.add_argument('--kernel_env', nargs="?", default='gauss', choices=['gauss', 'exp'],
                        help='Env Kernel choice')
    parser.add_argument('--kernel_env_param', nargs="?", default=None)
    # Experiment parameters
    parser.add_argument('--parallelization', nargs='?', type=bool, default=True)
    parser.add_argument('--max_horizon', nargs="?", type=int, default=10000, help='Maximum horizon')
    # State space
    parser.add_argument('--min_action', nargs="?", type=float, default=0)
    parser.add_argument('--max_action', nargs="?", type=float, default=1)
    parser.add_argument('--n_actions', nargs="?", type=int, default=10)
    parser.add_argument('--dim_actions', nargs="?", type=int, default=5)
    parser.add_argument('--min_context', nargs="?", type=float, default=0)
    parser.add_argument('--max_context', nargs="?", type=float, default=1)
    parser.add_argument('--n_contexts', nargs="?", type=int, default=10)
    parser.add_argument('--dim_contexts', nargs="?", type=int, default=5)
    parser.add_argument('--discrete_contexts', nargs='?', type=bool, default=True)
    # Random parameters
    parser.add_argument('--noise_scale', nargs="?", type=float, default=1)
    parser.add_argument('--rd_seeds_agent', nargs="+", type=float, default=[0, 1, 2, 3, 4], help='Random seeds Agent')
    parser.add_argument('--rd_seeds_env', nargs="+", type=float, default=[5, 6, 7, 8, 9], help='Random seed Env')
    parser.add_argument('--exp_name', nargs="?", type=str, default='default_exp', help='Name of the experiment')
    experiment(parser.parse_args())
