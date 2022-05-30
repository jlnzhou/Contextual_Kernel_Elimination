import os
import sys
from datetime import date

from tqdm import tqdm
import argparse

import numpy as np
import time

# Loaders
from src.loader import get_agent_by_name, get_env_by_name, get_kernel_by_name

today = date.today()

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


def save_result(settings, horizon, average_reward, regret, total_time):
    task_name = 'algo:{}'.format(settings['agent'])
    task_name += '|{}:{}'.format('mu', settings['mu'])
    task_name += '|{}:{}'.format('lambda', settings['reg_lambda'])
    task_name += '|{}:{}'.format('C', settings['C'])
    task_name += '|{}:{}'.format('beta', settings['beta'])
    task_name += '|{}:{}'.format('rd_seed', settings['random_seed'])
    task_name += '|{}:{}'.format('kernel', settings['kernel']),
    task_name += '|{}:{}'.format('env', settings['env']),
    task_name += '|{}:{}'.format('horizon', horizon)

    metrics_information = 'average_reward:{}'.format(average_reward)
    metrics_information += '|regret:{}'.format(regret)
    metrics_information += '|total_time:{}'.format(total_time)

    result = '{} {}\n'.format(task_name, metrics_information)
    results_dir = 'results/{}/{}/{}'.format(settings['env'], settings['exp_name'], today.strftime("%d-%m-%Y"))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_name = os.path.join(results_dir, 'metrics.txt')

    with open(file_name, 'a') as file:
        file.write(result)


def do_single_experiment(parameters):
    print('Running single experiment')
    # Instantiation
    env = get_env_by_name(parameters)
    kernel = get_kernel_by_name(parameters)
    agent = get_agent_by_name(parameters)(parameters, kernel)
    metrics = {'time': [],
               'average_reward': [],
               'regret': []
               }
    best_strategy_rewards = []

    # t0
    t0 = time.time()
    for step in tqdm(range(parameters['T'])):
        pass

    #
    #
    # t0 = time.time()
    #
    # for step in tqdm(range(parameters['T'] + 1)):
    #
    #     # choose a random context.
    #     context, label = env.sample_data()
    #     # iterate learning algorithm for 1 round.
    #     action = agent.sample_action(context)
    #     state = agent.get_state(context, action)
    #     reward = env.sample_reward_noisy(state, label)[0]
    #     agent.update_agent(context, action, reward)
    #     # get best_strategy's reward for the current context.
    #     best_strategy_rewards.append(env.get_best_reward_in_context(context, label))
    #
    #     if step % 100 == 0 and step != 0:
    #         t = time.time() - t0
    #         metrics['time'].append(t)
    #         average_reward = np.mean(agent.rewards[1:])
    #         metrics['average_reward'].append(average_reward)
    #         sum_best = np.sum(np.array(best_strategy_rewards))
    #         sum_agent = np.sum(np.array(agent.rewards[1:]))
    #         regret = sum_best - sum_agent
    #         save_result(parameters, step, average_reward, regret, t)
    #         print('Average reward: {}'.format(average_reward))
    #         print('Regret: {}'.format(regret))
    #         print('Dictionary size: {}'.format(agent.dictionary_size()))
    #
    # return metrics


def experiment(args):
    parameters = {
        # Algorithm
        'agent': args.algo,
        'kernel': args.kernel,
        'kernel_param': args.kernel_param,
        'reg_lambda': args.lbd,
        'mu': args.mu,
        # Environment
        'env': args.env,
        'T': args.max_horizon,
        'min_action': args.min_action,
        'max_action': args.max_action,
        'n_actions': args.n_actions,
        'dim_actions': args.dim_actions,
        'min_context': args.min_context,
        'max_context': args.max_context,
        'n_contexts': args.n_contexts,
        'dim_contexts': args.dim_contexts,
        'noise_scale': args.noise_scale,
        # Experiment parameters
        'exp_name': args.exp_name
    }
    print(parameters)
    for rd in args.rd_seeds:
        parameters['random_seed'] = rd
        print("Random seed: {}".format(rd))
        do_single_experiment(parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    # Algorithm
    parser.add_argument('--algo', nargs="?", default='k_ucb', choices=['k_ucb'], help='Algorithm')
    parser.add_argument('--kernel', nargs="?", default='gauss', choices=['gauss', 'exp'], help='Kernel choice')
    parser.add_argument('--kernel_param', nargs="?", default=None)
    parser.add_argument('--lbd', nargs="?", type=float, default=1, help='Regularization parameter')
    parser.add_argument('--mu', nargs="?", type=float, default=1, help='Projection parameter')
    # Environment
    parser.add_argument('--env', nargs="?", default='bump', choices=['bump'], help='Environment')
    parser.add_argument('--max_horizon', nargs="?", type=int, default=1000, help='Maximum horizon')
    parser.add_argument('--min_action', nargs="?", type=float, default=0)
    parser.add_argument('--max_action', nargs="?", type=float, default=1)
    parser.add_argument('--n_actions', nargs="?", type=float, default=101)
    parser.add_argument('--dim_actions', nargs="?", type=int, default=2)
    parser.add_argument('--min_context', nargs="?", type=float, default=0)
    parser.add_argument('--max_context', nargs="?", type=float, default=1)
    parser.add_argument('--n_contexts', nargs="?", type=float, default=101)
    parser.add_argument('--dim_contexts', nargs="?", type=int, default=5)
    parser.add_argument('--noise_scale', nargs="?", type=float, default=0.1)
    # Experiment parameters
    parser.add_argument('--rd_seeds', nargs="+", type=float, default=[1, 2, 3], help='Random seed')
    parser.add_argument('--exp_name', nargs="?", type=str, default='exp', help='Name of the experiment')
    experiment(parser.parse_args())
