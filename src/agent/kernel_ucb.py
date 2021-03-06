import jax.numpy as jnp
from jax.config import config
from jax import vmap

from src.agent.agent import Agent

config.update("jax_enable_x64", True)


class KernelUCBDiscrete(Agent):

    def __init__(self,
                 settings,
                 kernel):
        """
        Initializes the class
        """
        super(KernelUCBDiscrete, self).__init__(settings)
        self.reg_lambda = settings['reg_lambda']
        self.explo = settings["explo"]
        self.kernel = kernel
        self.matrix_kt = None
        self.matrix_kt_inverse = None

    # Sampling
    def get_upper_confidence_bound(self, state):
        k_past_present = self.kernel.evaluate(self.past_states, state)
        mean = jnp.dot(k_past_present.T, jnp.dot(self.matrix_kt_inverse, self.rewards))
        k_present_present = self.kernel.evaluate(state, state)
        std2 = k_present_present - jnp.dot(k_past_present.T, jnp.dot(self.matrix_kt_inverse,
                                                                     k_past_present))
        ucb = mean + self.explo/jnp.sqrt(self.reg_lambda) * jnp.sqrt(std2)
        return jnp.squeeze(ucb)

    def discrete_inference(self, states_grid):
        if self.past_states.size == 0:
            return self.agent_rng.choice(a=self.actions,
                                         size=self.actions_dim)
        else:
            ucb_all_actions = vmap(self.get_upper_confidence_bound)(states_grid)
            idx = jnp.argmax(ucb_all_actions)
            state = states_grid[idx]
            context, action = state[:, :self.contexts_dim], state[:, self.contexts_dim:]
            return action

    def sample_action(self, context):
        return self.discrete_inference(context)

    # Updating agent
    def update_agent(self, state, reward_clean, reward):
        self.update_data_pool(state, reward_clean, reward)
        k_present_present = self.kernel.evaluate(state, state)
        if self.past_states.size == self.actions_dim+self.contexts_dim:
            self.matrix_kt_inverse = 1/k_present_present + self.reg_lambda
        else:
            # Update inverse gram matrix online
            k_past_present = self.kernel.evaluate(self.past_states, state)[:-1]
            k22 = k_present_present + self.reg_lambda - jnp.dot(k_past_present.T,
                                                                jnp.dot(self.matrix_kt_inverse, k_past_present))
            k22 = 1/k22
            k11 = jnp.dot(self.matrix_kt_inverse, k_past_present)
            k11 = jnp.dot(k11, k11.T)
            k11 = self.matrix_kt_inverse + k22*k11
            k12 = -k22*jnp.dot(self.matrix_kt_inverse, k_past_present)
            k21 = -k22*jnp.dot(k_past_present.T, self.matrix_kt_inverse)
            self.matrix_kt_inverse = jnp.block([[k11, k12],
                                                [k21, k22]])
