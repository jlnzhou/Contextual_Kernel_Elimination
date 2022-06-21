import os
import sys
from numpy import random
from jax.config import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


class Env:

    def __init__(self,
                 parameters):
        # Horizon
        self.horizon = parameters['T']
        # Contexts/Actions parameters
        self.actions = jnp.array(parameters['actions'])
        self.contexts = jnp.array(parameters['contexts'])
        self.actions_dim = parameters['dim_actions']
        self.contexts_dim = parameters['dim_contexts']
        # RNG parameters
        self.random_seed = parameters['random_seed_env']
        self.env_rng = random.RandomState(self.random_seed)
        self.noise_scale = parameters['noise_scale']

    def sample_data(self):
        pass

    def sample_reward(self, state):
        pass

    def sample_reward_noisy(self, state):
        pass

    def get_best_reward_in_context(self, context, states_grid):
        pass
