import jax.numpy as jnp
from jax.config import config
import jax as jax
import numpy as np
config.update("jax_enable_x64", True)

default_gaussian_scale = 0.1
default_exp_scale = 10


@jax.jit
def sq_euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)


# RBF Kernel
@jax.jit
def rbf_kernel(gamma, x, y):
    return jnp.exp(- gamma * sq_euclidean_distance(x, y))


# Exponential Kernel
@jax.jit
def exp_kernel(gamma, x, y):
    return jnp.exp(- gamma * jnp.sqrt(sq_euclidean_distance(x, y)))


def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


class Kernel:

    def __init__(self,
                 settings):
        """Initializes the class
        """
        self.settings = settings

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, state1, state2):
        pass


class Gaussian(Kernel):

    def __init__(self,
                 *args,
                 gaussian_scale):
        """Initializes the class
        """
        super(Gaussian, self).__init__(*args)
        self.gaussian_scale = gaussian_scale
        if self.gaussian_scale is None:
            self.gaussian_scale=default_gaussian_scale

    def _pairwise(self, state1, state2):
        """
        Args:
            state1 (np.ndarray)
            state2 (np.ndarray)
        """
        return gram(rbf_kernel, 1 / (2 * self.gaussian_scale ** 2), state1, state2)


class Exponential(Kernel):

    def __init__(self,
                 *args,
                 exp_scale):
        """Initializes the class
        """
        super(Exponential, self).__init__(*args)
        """Initializes the class
        """
        self.exp_scale = exp_scale
        if self.exp_scale is None:
            self.exp_scale = default_exp_scale

    def _pairwise(self, state1, state2):
        """
        Args:
            state1 (np.ndarray)
            state2 (np.ndarray)
        """
        return gram(exp_kernel, self.exp_scale, state1, state2)
