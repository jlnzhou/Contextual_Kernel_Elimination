import os
import sys
from jax.config import config
import numpy as np
# Agents
from src.agent.kernel_ucb import KernelUCB
# Kernels
from src.kernel import Gaussian, Exponential
# Environments
from src.env.env_bump import EnvBump

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


def get_agent_by_name(parameters):
    if parameters['agent'] == 'k_ucb':
        return KernelUCB
    else:
        raise NotImplementedError


def get_env_by_name(parameters):
    if parameters['env'] == 'bump':
        return EnvBump(rd_seed=parameters['random_seed_env'],
                       horizon=parameters['T'],
                       actions=np.linspace(parameters['min_action'], parameters["max_action"], parameters["n_actions"]),
                       contexts=np.linspace(parameters['min_context'], parameters["max_context"], parameters["n_contexts"]),
                       actions_dim=parameters["dim_actions"],
                       contexts_dim=parameters["dim_contexts"],
                       noise_scale=parameters["noise_scale"])
    else:
        raise NotImplementedError


def get_kernel_by_name(parameters):
    if parameters['kernel'] == 'gauss':
        return Gaussian(parameters,
                        gaussian_scale=parameters['kernel_param'])
    elif parameters['kernel'] == 'exp':
        return Exponential(parameters,
                           exp_scale=parameters['kernel_params'])
    else:
        raise NotImplementedError
