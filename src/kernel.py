import os
import sys
import jax.numpy as jnp
from jax.config import config
from jax import jit, vmap


config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

default_gaussian_scale = 0.1
default_exp_scale = 10


@jit
def sq_euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)


# RBF Kernel
@jit
def rbf_kernel(gamma, x, y):
    return jnp.exp(- gamma * sq_euclidean_distance(x, y))


# Exponential Kernel
@jit
def exp_kernel(gamma, x, y):
    return jnp.exp(- gamma * jnp.sqrt(sq_euclidean_distance(x, y)))


def gram(func, params, x, y):
    return vmap(lambda x1: vmap(lambda y1: func(params, x1, y1))(y))(x)


class Kernel:

    def __init__(self,
                 settings, type):
        """Initializes the class
        """
        self.settings = settings
        self.type = type

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, state1, state2):
        pass


class Gaussian(Kernel):

    def __init__(self,
                 parameters,
                 type):
        """Initializes the class
        """
        self.gaussian_scale = None
        super(Gaussian, self).__init__(parameters, type)
        if self.type == 'env':
            self.gaussian_scale = parameters["kernel_env_param"]
        elif self.type == 'agent':
            self.gaussian_scale = parameters["kernel_agent_param"]
        else:
            pass
        if self.gaussian_scale is None:
            self.gaussian_scale = default_gaussian_scale

    def _pairwise(self, state1, state2):
        """
        Args:
            state1 (np.ndarray)
            state2 (np.ndarray)
        """
        return gram(rbf_kernel, 1 / (2 * self.gaussian_scale ** 2), state1, state2)


class Exponential(Kernel):

    def __init__(self,
                 parameters, type):
        """Initializes the class
        """
        super(Exponential, self).__init__(parameters, type)
        """Initializes the class
        """
        if self.type == 'env':
            self.gaussian_scale = parameters["kernel_env_param"]
        elif self.type == 'agent':
            self.gaussian_scale = parameters["kernel_agent_param"]
        else:
            pass
        self.exp_scale = parameters['kernel_param']
        if self.exp_scale is None:
            self.exp_scale = default_exp_scale

    def _pairwise(self, state1, state2):
        """
        Args:
            state1 (np.ndarray)
            state2 (np.ndarray)
        """
        return gram(exp_kernel, self.exp_scale, state1, state2)
