import jax.numpy as jnp
from jax.config import config
from jax import vmap

from src.agent.agent import Agent


config.update("jax_enable_x64", True)


class AgentRandomDiscrete(Agent):

    def __init__(self,
                 settings,
                 kernel):
        """
        Initializes the class
        """
        super(AgentRandomDiscrete, self).__init__(settings)
        self.kernel = kernel

    # Random Action
    def sample_action(self, context):
        return self.agent_rng.choice(a=self.actions,
                                     size=self.actions_dim)

    def update_agent(self, state, reward_clean, reward):
        self.update_data_pool(state, reward_clean, reward)
