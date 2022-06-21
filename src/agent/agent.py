import os
import sys
from numpy import random
from jax.config import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


class Agent:

    def __init__(self,
                 parameters):

        self.settings = parameters
        # RNG parameters
        self.random_seed = parameters['random_seed_agent']
        self.agent_rng = random.RandomState(self.random_seed)
        self.noise_scale = parameters['noise_scale']
        # Actions
        self.actions = jnp.array(self.settings['actions'])
        self.actions_dim = self.settings['dim_actions']
        self.actions_grid = self.settings['actions_grid']
        # Contexts
        self.contexts = self.settings['contexts']
        self.contexts_dim = self.settings['dim_contexts']
        # History
        self.past_states = jnp.array([]).reshape(0, self.contexts_dim + self.actions_dim)
        self.rewards_clean = jnp.array([]).reshape(0)
        self.rewards = jnp.array([]).reshape(0)

    def sample_action(self, context):
        pass

    def update_data_pool(self, state, reward_clean, reward):
        self.past_states = jnp.concatenate([self.past_states, state])
        self.rewards_clean = jnp.concatenate([self.rewards_clean, reward_clean])
        self.rewards = jnp.concatenate([self.rewards, reward])

    def update_agent(self, state, reward_clean, reward):
        pass
