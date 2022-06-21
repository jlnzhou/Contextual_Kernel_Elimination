import os
import sys
from jax.config import config

from src.agent.kernel_ucb import KernelUCBDiscrete
from src.kernel import Gaussian, Exponential
from src.env.env_bump import EnvBumpDiscrete
from src.env.env_kernel_linear import EnvKernelLinearDiscrete

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)
config.update("jax_enable_x64", True)


def get_env_by_name(parameters, kernel):
    if parameters['env'] == 'bump' and parameters['discrete_contexts']:
        return EnvBumpDiscrete(parameters)
    elif parameters['env'] == 'kernel_linear' and parameters['discrete_contexts']:
        return EnvKernelLinearDiscrete(parameters, kernel)
    else:
        raise NotImplementedError


def get_kernel_by_name(parameters, type):
    kernel = None
    if type == 'agent':
        kernel = parameters["kernel_agent"]
    elif type == 'env':
        kernel = parameters['kernel_env']
    else:
        raise NotImplementedError
    if kernel == 'gauss':
        return Gaussian(parameters, type)
    elif parameters['kernel'] == 'exp':
        return Exponential(parameters, type)
    else:
        raise NotImplementedError


def get_agent_by_name(parameters):
    if parameters['agent'] == 'k_ucb' and parameters['discrete_contexts']:
        return KernelUCBDiscrete
    else:
        raise NotImplementedError
