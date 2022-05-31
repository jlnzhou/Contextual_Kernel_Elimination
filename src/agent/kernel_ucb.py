import itertools

import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
from jax import grad, hessian
import numpy as np
from profilehooks import timecall

config.update("jax_enable_x64", True)

default_beta = 0.1


class KernelUCB:

    def __init__(self,
                 settings,
                 kernel,
                 beta_t=default_beta):
        """
        Initializes the class
        """
        self.settings = settings
        self.agent_rng = np.random.RandomState(self.settings['random_seed_agent'])
        self.reg_lambda = settings['reg_lambda']
        self.kernel = kernel
        # Actions
        self.actions = np.linspace(settings['min_action'], settings["max_action"], settings["n_actions"])
        self.actions_dim = settings['dim_actions']
        self.actions_grid = [[self.actions[i] for i in a] for a in itertools.product(range(len(self.actions)), repeat=self.actions_dim)]
        # Contexts
        self.contexts = np.linspace(settings['min_context'], settings["max_context"], settings["n_contexts"])
        self.contexts_dim = settings['dim_contexts']
        # Other attributes
        self.past_states = jnp.array([]).reshape(0, self.contexts_dim+self.actions_dim)
        self.rewards = jnp.array([]).reshape(0)
        self.matrix_kt = None
        self.matrix_kt_inverse = None
        self.beta_t = beta_t

    # Sampling
    def get_upper_confidence_bound(self, state):
        k_past_present = self.kernel.evaluate(self.past_states, state)
        mean = jnp.dot(k_past_present.T, jnp.dot(self.matrix_kt_inverse, self.rewards))
        k_present_present = self.kernel.evaluate(state, state)
        std2 = (1 / self.reg_lambda) * (k_present_present - jnp.dot(k_past_present.T,
                                                                    jnp.dot(self.matrix_kt_inverse,
                                                                            k_past_present)))
        ucb = mean + self.beta_t * jnp.sqrt(std2)
        return jnp.squeeze(ucb)

    def get_batch_ucb(self, context):
        return

    def discrete_inference(self, context):
        if self.past_states.size == 0:
            return self.agent_rng.choice(a=self.actions,
                                         size=self.actions_dim)
        else:
            states_grid = [self.get_state(context, jnp.array(a)) for a in self.actions_grid]
            ucb_all_actions = jnp.array([self.get_upper_confidence_bound(s) for s in states_grid])
            idx = jnp.argmax(ucb_all_actions)
            return jnp.array(self.actions_grid[idx])

    def sample_action(self, context):
        return self.discrete_inference(context)

    @staticmethod
    def get_state(context, action):
        context, action = context.reshape((1, -1)), action.reshape((1, -1))
        return jnp.concatenate([context, action], axis=1)

    # Updating agent
    def update_data_pool(self, context, action, reward):
        state = self.get_state(context, action)
        self.past_states = jnp.concatenate([self.past_states, state])
        self.rewards = jnp.concatenate([self.rewards, reward])

    def set_gram_matrix(self):
        self.matrix_kt = self.kernel.gram_matrix(self.past_states)
        self.matrix_kt += self.reg_lambda * jnp.eye(self.matrix_kt.shape[0])
        self.matrix_kt_inverse = jnp.linalg.inv(self.matrix_kt)

    def update_agent(self, context, action, reward):
        self.update_data_pool(context, action, reward)
        self.set_gram_matrix()







