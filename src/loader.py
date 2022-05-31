import os
import sys
from jax.config import config

from src.agent.kernel_ucb import KernelUCBDiscrete
from src.kernel import Gaussian, Exponential
from src.env.env_bump import EnvBumpDiscrete

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)
config.update("jax_enable_x64", True)


def get_env_by_name(parameters):
    if parameters['env'] == 'bump' and parameters['discrete_contexts']:
        return EnvBumpDiscrete(parameters)
    else:
        raise NotImplementedError


def get_kernel_by_name(parameters):
    if parameters['kernel'] == 'gauss':
        return Gaussian(parameters)
    elif parameters['kernel'] == 'exp':
        return Exponential(parameters)
    else:
        raise NotImplementedError


def get_agent_by_name(parameters):
    if parameters['agent'] == 'k_ucb' and parameters['discrete_contexts']:
        return KernelUCBDiscrete
    else:
        raise NotImplementedError
