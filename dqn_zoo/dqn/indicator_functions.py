
import functools
import operator
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union

import chex
from flax import core
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from dqn_zoo import networks


@chex.dataclass
class IndicatorOutput:
  rewards: chex.Array
  pre_threshold: Optional[chex.Array] = None


class Indicator(nn.Module):
  """A network comprised of a stack of indictor networks."""
  num_auxiliary_tasks: int = 10
  width_multiplier: float = 1.0
  tasks_per_module: int = 1
  apply_final_relu: bool = True
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, obs):
    obs = obs.astype(self.input_dtype)
    # outputs = networks.ConvNet(self.num_auxiliary_tasks, name='encoder')(obs)
    outputs = networks.NatureDQNNetwork(
      self.num_auxiliary_tasks, name='encoder')(obs).predictions

    outputs = jax.lax.stop_gradient(outputs)

    reward_bias = self.param('reward_bias', nn.initializers.zeros,
                             (self.num_auxiliary_tasks,), self.param_dtype)
    outputs = outputs + reward_bias

    rewards = jnp.where(outputs <= 0.0, 0.0, 1.0)
    rewards = jax.lax.stop_gradient(rewards)

    return IndicatorOutput(pre_threshold=outputs, rewards=rewards)


class TrainState(struct.PyTreeNode):
  """Train State. This resembles Flax's train state."""
  step: int
  apply_fn: Callable[Ellipsis, Any] = struct.field(pytree_node=False)
  params: chex.ArrayTree
  optim: optax.GradientTransformation = struct.field(pytree_node=False)
  optim_state: optax.OptState

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_optim_state = self.optim.update(grads, self.optim_state,
                                                 self.params)
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        optim_state=new_optim_state,
        **kwargs)

  @classmethod
  def create(cls, *, apply_fn, params,
             optim, **kwargs):
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params))


def create_indicator_state(
    dummy_obs,
    rng,
    lr,
    lap_dim
):
  model = Indicator(num_auxiliary_tasks=lap_dim)
  optim = optax.adam(lr)
  optim_mask = {'params': {'encoder': False, 'reward_bias': True}}
  optim = optax.masked(optim, optim_mask)

  params = model.init(rng, dummy_obs)
  # We have to unfreeze the parameters or else optax has some issues
  # with optax.masked
  params = params.unfreeze()

  indicator_state = TrainState.create(
      apply_fn=model.apply, params=params, optim=optim)

  return indicator_state
