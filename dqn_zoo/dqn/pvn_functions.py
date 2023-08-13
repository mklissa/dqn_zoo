
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

TargetParamsUpdateFunction = Callable[[chex.ArrayTree, chex.ArrayTree, int],
                                      chex.ArrayTree]

def identity(buffer, dummy_variable):
  """Identity fn. with non-trivial computation to prevent jit optimization."""
  return buffer, jnp.sin(dummy_variable**2)


@chex.dataclass
class PVNOutput:
  phi: chex.Array
  predictions: Optional[chex.Array] = None


class PVNetwork(nn.Module):
  num_auxiliary_tasks: int = 10
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, obs):

    obs = obs.astype(self.input_dtype)
    # predictions = networks.ConvNet(self.num_auxiliary_tasks, name='aux_tasks')(obs)
    pvn_output = networks.NatureDQNNetwork(
      self.num_auxiliary_tasks, name='aux_tasks')(obs)

    return PVNOutput(phi=pvn_output.phi, predictions=pvn_output.predictions)


class FittedValueTrainState(struct.PyTreeNode):
  """Train State for fitted value iteration methods."""
  step: int
  apply_fn: Callable[Ellipsis, Any] = struct.field(pytree_node=False)
  params: chex.ArrayTree
  optim: optax.GradientTransformation = struct.field(pytree_node=False)
  optim_state: optax.OptState
  target_params: core.FrozenDict[str, Any]
  target_params_update_fn: TargetParamsUpdateFunction = struct.field(
      pytree_node=False)

  def apply_gradients(self, *, grads,
                      **kwargs):
    updates, new_optim_state = self.optim.update(grads, self.optim_state,
                                                 self.params)
    new_params = optax.apply_updates(self.params, updates)
    new_target_params = self.target_params_update_fn(self.params,
                                                     self.target_params,
                                                     self.step)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        target_params=new_target_params,
        optim_state=new_optim_state,
        **kwargs)

  @classmethod
  def create(cls, *, apply_fn, params,
             target_params_update_fn,
             optim):
    target_params = operator.getitem(jax.jit(identity)(params, 2), 0)

    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params),
        target_params=target_params,
        target_params_update_fn=target_params_update_fn)


def construct_soft_target_params_update_fn(tau):
  def wrapper(new_params, old_params,
              unused_step):
    # The current step is unused as we just EMA the params.
    ema = lambda new, old: (1.0 - tau) * new + tau * old
    return jax.tree_map(ema, new_params, old_params)

  return wrapper


def create_train_state(
    dummy_obs,
    rng,
    lr,
    lap_dim
):
  target_params_update_fn = construct_soft_target_params_update_fn(0.99)
  model = PVNetwork(num_auxiliary_tasks=lap_dim)
  optim = optax.adam(lr)
  params = model.init(rng, dummy_obs)

  return FittedValueTrainState.create(
      apply_fn=model.apply,
      params=params,
      target_params_update_fn=target_params_update_fn,
      optim=optim)


def get_pvn(pvn_state, pvn_params, obs):
    return pvn_state.apply_fn(pvn_params, obs)
