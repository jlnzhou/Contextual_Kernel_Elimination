import os
import sys
import jax.numpy as jnp
from jax.config import config

from src.env.env import Env

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


class EnvBumpDiscrete(Env):

    def __init__(self,
                 parameters):
        super(EnvBumpDiscrete, self).__init__(parameters)
        # Star Parameters
        self.a_star = self.env_rng.choice(a=self.actions,
                                          size=self.actions_dim)
        self.x_star = self.env_rng.choice(a=self.contexts,
                                          size=self.contexts_dim)
        self.w_star = self.env_rng.choice(a=self.contexts,
                                          size=self.contexts_dim)

    def sample_data(self):
        return self.env_rng.choice(a=self.contexts,
                                   size=self.contexts_dim)

    def sample_reward(self, state):
        context, action = state[:, :self.contexts_dim], state[:, self.contexts_dim:]
        term = jnp.linalg.norm(action - self.a_star, ord=1) + jnp.dot(context - self.x_star, self.w_star).squeeze()
        r = max(0, 1 - term)
        return jnp.array([r])

    def sample_reward_noisy(self, reward):
        return reward + self.env_rng.normal(loc=0.0, scale=self.noise_scale)

    def get_best_reward_in_context(self, context, states_grid):
        term = jnp.dot(context - self.x_star, self.w_star).squeeze()
        r = max(0, 1 - term)
        return jnp.array([r])
