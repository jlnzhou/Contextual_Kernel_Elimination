import os
import sys
from jax.config import config
# Agents
from src.agent.kernel_ucb import KernelUCB
# Kernels
from src.kernel import Gaussian, Exponential
# Environments
from src.env.env_bump import EnvBump

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


def get_agent_by_name(settings):
    if settings['agent'] == 'k_ucb':
        return KernelUCB
    else:
        raise NotImplementedError


def get_env_by_name(settings):
    if settings['env'] == 'bump':
        return EnvBump
    else:
        raise NotImplementedError


def get_kernel_by_name(settings):
    if settings['kernel'] == 'gauss':
        return Gaussian
    elif settings['kernel'] == 'exp':
        return Exponential
    else:
        raise NotImplementedError
