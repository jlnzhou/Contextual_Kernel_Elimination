import os
import sys
import jax.numpy as jnp
from jax import vmap
from jax.config import config

from src.env.env import Env
from src.utils import get_state

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


class EnvKernelLinearDiscrete(Env):

    def __init__(self,
                 parameters,
                 kernel):
        super(EnvKernelLinearDiscrete, self).__init__(parameters)
        self.x_star = self.env_rng.choice(a=self.contexts,
                                          size=self.contexts_dim)
        self.a_star = self.env_rng.choice(a=self.actions,
                                          size=self.actions_dim)
        self.theta_star = get_state(self.x_star, self.a_star)
        self.kernel = kernel

    def sample_data(self):
        return self.env_rng.choice(a=self.contexts,
                                   size=self.contexts_dim)

    def sample_reward(self, state):
        result = self.kernel.evaluate(self.theta_star, state).squeeze()
        return jnp.array([result])

    def sample_reward_noisy(self, reward):
        return reward + self.env_rng.normal(loc=0.0, scale=self.noise_scale)

    def get_best_reward_in_context(self, context, states_grid):
        rewards = vmap(self.sample_reward)(states_grid)
        result = jnp.max(rewards).squeeze()
        return jnp.array([result])
