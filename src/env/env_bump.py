import numpy as np


class EnvBump:

    def __init__(self,
                 rd_seed,
                 horizon,
                 actions,
                 contexts,
                 actions_dim,
                 contexts_dim,
                 noise_scale):
        self.random_seed = rd_seed
        # Horizon
        self.horizon = horizon
        # Contexts/Actions parameters
        self.actions = actions
        self.contexts = contexts
        self.input_mesh = np.array(np.meshgrid(actions, contexts))
        self.context_numbers = contexts.size
        self.actions_dim = actions_dim
        self.contexts_dim = contexts_dim
        # RNG parameters
        self.env_rng = np.random.RandomState(rd_seed)  # np.random.RandomState(123)
        self.noise_scale = noise_scale
        # Star Parameters
        self.a_star = self.env_rng.choice(a=self.actions,
                                          size=self.actions_dim)
        self.x_star = self.env_rng.choice(a=self.contexts,
                                          size=self.contexts_dim)
        self.w_star = self.env_rng.choice(a=self.contexts,
                                          size=self.contexts_dim)

    def sample_reward(self, state):
        context, action = state[:, self.contexts_dim], state[:, self.contexts_dim:]
        term = np.linalg.norm(action - self.a_star, ord=1) + np.dot(context - self.x_star, self.w_star).squeeze()
        r = max(0, 1 - term)
        return np.array([r])

    def sample_reward_noisy(self, state):
        return [self.sample_reward(state) + self.env_rng.normal(loc=0.0, scale=self.noise_scale)]

    def find_best_input_in_joint_space(self, joint_pair):
        return np.argmax(self.sample_reward(joint_pair))

    def get_best_reward_in_context(self, context):
        term = np.dot(context - self.x_star, self.w_star).squeeze()
        r = max(0, 1 - term)
        return r
