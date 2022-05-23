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
    results_dir = 'results/{}/{}/{}'.format(settings['env'], settings['expname'], today.strftime("%d-%m-%Y"))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_name = os.path.join(results_dir, 'metrics.txt')

    with open(file_name, 'a') as file:
        file.write(result)


def instantiate_metrics():
    return {
        'time': [],
        'average_reward': [],
        'regret': [],
    }


def do_single_experiment(parameters):
    print('Env: {}'.format(parameters['env']))
    print('Running experiment with agent {}, lbd {}, mu {}, beta {}, C {}, rd {}'.format(parameters['agent'],
                                                                                         parameters['reg_lambda'],
                                                                                         parameters['mu'],
                                                                                         parameters['beta'],
                                                                                         parameters['C'],
                                                                                         parameters['random_seed']))
    env = get_env_by_name(parameters)(parameters['random_seed'])
    kernel = get_kernel_by_name(parameters)(parameters)
    agent = get_agent_by_name(parameters)(parameters, kernel)
    agent.instantiate(env)
    metrics = instantiate_metrics()
    best_strategy_rewards = []

    if env.horizon:
        parameters['T'] = env.horizon

    t0 = time.time()

    for step in tqdm(range(parameters['T'] + 1)):

        # choose a random context.
        context, label = env.sample_data()
        # iterate learning algorithm for 1 round.
        action = agent.sample_action(context)
        state = agent.get_state(context, action)
        reward = env.sample_reward_noisy(state, label)[0]
        agent.update_agent(context, action, reward)
        # get best_strategy's reward for the current context.
        best_strategy_rewards.append(env.get_best_reward_in_context(context, label))

        if step % 100 == 0 and step != 0:
            t = time.time() - t0
            metrics['time'].append(t)
            average_reward = np.mean(agent.rewards[1:])
            metrics['average_reward'].append(average_reward)
            sum_best = np.sum(np.array(best_strategy_rewards))
            sum_agent = np.sum(np.array(agent.rewards[1:]))
            regret = sum_best - sum_agent
            save_result(parameters, step, average_reward, regret, t)
            print('Average reward: {}'.format(average_reward))
            print('Regret: {}'.format(regret))
            print('Dictionary size: {}'.format(agent.dictionary_size()))

    return metrics


def experiment(args):
    for rd in args.rd_seeds:
        parameters = {
            'agent': args.algo,
            'kernel': args.kernel,
            'reg_lambda': args.lbd,
            'T': args.max_horizon,
            'env': args.env,
            'mu': args.mu,
            'random_seed': rd,
            'exp_name': args.exp_name
        }
        print(parameters)
        do_single_experiment(parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--algo', nargs="?", default='k_ucb', choices=['k_ucb'],
                        help='Algorithm')
    parser.add_argument('--kernel', nargs="?", default='gauss', choices=['gauss', 'exp'],
                        help='Kernel choice')
    parser.add_argument('--lbd', nargs="?", type=float, default=1, help='Regularization parameter')
    parser.add_argument('--max_horizon', nargs="?", type=int, default=1000, help='Maximum horizon')
    parser.add_argument('--env', nargs="?", default='bump', choices=['bump'],
                        help='Environment')
    parser.add_argument('--mu', nargs="?", type=float, default=1, help='Projection parameter')
    parser.add_argument('--rd_seeds', nargs="+", type=float, default=[1, 2, 3], help='Random seed')
    parser.add_argument('--exp_name', nargs="?", type=str, default='experiment', help='Name of the experiment')
    experiment(parser.parse_args())
