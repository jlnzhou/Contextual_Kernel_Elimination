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


def save_result(settings, metrics):
    task_name = 'algo:{}'.format(settings['agent'])
    task_name += '|{}:{}'.format('kernel', settings['kernel'])
    task_name += '|{}:{}'.format('kernel_param', settings['kernel_param'])
    task_name += '|{}:{}'.format('reg_lambda', settings['reg_lambda'])
    task_name += '|{}:{}'.format('mu', settings['mu'])
    task_name += '|{}:{}'.format('env', settings['env'])
    task_name += '|{}:{}'.format('T', settings['T'])
    task_name += '|{}:{}'.format('min_action', settings['min_action'])
    task_name += '|{}:{}'.format('max_action', settings['max_action'])
    task_name += '|{}:{}'.format('n_actions', settings['n_actions'])
    task_name += '|{}:{}'.format('dim_actions', settings['dim_actions'])
    task_name += '|{}:{}'.format('min_context', settings['min_context'])
    task_name += '|{}:{}'.format('max_context', settings['max_context'])
    task_name += '|{}:{}'.format('n_contexts', settings['n_contexts'])
    task_name += '|{}:{}'.format('dim_contexts', settings['dim_contexts'])
    task_name += '|{}:{}'.format('noise_scale', settings['noise_scale'])
    task_name += '|{}:{}'.format('exp_name', settings['exp_name'])
    task_name += '|{}:{}'.format('random_seed_agent', settings['random_seed_agent'])
    task_name += '|{}:{}'.format('random_seed_env', settings['random_seed_env'])

    metrics_information = 'step:{}'.format(metrics['step'])
    metrics_information += '|time:{}'.format(metrics['time'])
    metrics_information += '|average_reward:{}'.format(metrics['average_reward'])
    metrics_information += '|average_best:{}'.format(metrics['average_best'])
    metrics_information += '|sum_reward:{}'.format(metrics['sum_reward'])
    metrics_information += '|sum_best:{}'.format(metrics['sum_best'])
    metrics_information += '|regret:{}'.format(metrics['regret'])

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
    metrics = {'step': [],
               'time': [],
               'average_reward': [],
               'average_best': [],
               'sum_reward': [],
               'sum_best': [],
               'regret': []
               }
    best_strategy_rewards = []

    # t0
    t0 = time.time()

    # Iterations
    for step in tqdm(range(parameters['T'])):
        # Choose a random context
        context = env.sample_data()
        # Iteration of the agent
        action = agent.sample_action(context)
        state = agent.get_state(context, action)
        reward = env.sample_reward_noisy(state)
        agent.update_agent(context, action, reward)
        # Best reward possible
        best_strategy_rewards.append(env.get_best_reward_in_context(context).squeeze())

        # Metrics
        if step % 100 == 0 and step != 0:
            metrics['step'] = step
            t = time.time() - t0
            metrics['time'].append(t)
            average_reward = np.mean(agent.rewards)
            metrics['average_reward'].append(average_reward)
            average_best = np.mean(best_strategy_rewards)
            metrics['average_best'].append(average_best)
            sum_best = np.sum(best_strategy_rewards)
            metrics['sum_best'].append(sum_best)
            sum_agent = np.sum(agent.rewards)
            metrics['sum_reward'].append(sum_agent)
            regret = sum_best - sum_agent
            metrics['regret'].append(regret)
            save_result(parameters, metrics)
            print('Step: {}'.format(step))
            print('Average reward: {}'.format(average_reward))
            print('Regret: {}'.format(regret))


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
    for (rd_agent, rd_env) in zip(args.rd_seeds_agent, args.rd_seeds_env):
        parameters['random_seed_agent'] = rd_agent
        parameters['random_seed_env'] = rd_env
        print("Random seed Agent: {}".format(rd_agent))
        print("Random seed Env: {}".format(rd_env))
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
    parser.add_argument('--rd_seeds_agent', nargs="+", type=float, default=[1, 2, 3], help='Random seeds Agent')
    parser.add_argument('--rd_seeds_env', nargs="+", type=float, default=[4, 5, 6], help='Random seed Env')
    parser.add_argument('--exp_name', nargs="?", type=str, default='exp', help='Name of the experiment')
    experiment(parser.parse_args())
