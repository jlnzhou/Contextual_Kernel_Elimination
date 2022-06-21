import os
import sys

import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


def get_state(context, action):
    context, action = context.reshape((1, -1)), action.reshape((1, -1))
    return jnp.concatenate([context, action], axis=1)