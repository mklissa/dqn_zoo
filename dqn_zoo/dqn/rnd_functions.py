
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
class RNDOutput:
  predictions: chex.Array
  targets: Optional[chex.Array]
  prediction_error: chex.Array


class RND(nn.Module):
  num_auxiliary_tasks: int = 512
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, obs):

    obs = obs.astype(self.input_dtype)

    predictions = networks.ConvNet(
        self.num_auxiliary_tasks, name='predictions')(obs)

    targets = networks.ConvNet(
        self.num_auxiliary_tasks, name='targets')(obs)

    # predictions = networks.NatureDQNNetwork(
    #     self.num_auxiliary_tasks, 1., name='predictions')(obs).predictions

    # targets = networks.NatureDQNNetwork(
    #     self.num_auxiliary_tasks, 1., name='targets')(obs).predictions

    targets = jax.lax.stop_gradient(targets)

    prediction_error = ((targets - predictions)**2).mean(axis=1)

    return RNDOutput(
        predictions=predictions,
        targets=targets,
        prediction_error=prediction_error)


class RNDState(struct.PyTreeNode):
  """Train State. This resembles Flax's train state."""
  step: int
  apply_fn: Callable[Ellipsis, Any] = struct.field(pytree_node=False)
  params: chex.ArrayTree
  optim: optax.GradientTransformation = struct.field(pytree_node=False)
  optim_state: optax.OptState
  # batch_stats: Any

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
             optim, 
            #  batch_stats,
             **kwargs):
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params),
        # batch_stats=batch_stats
        )


def create_rnd_state(dummy_obs, rng, optim, lap_dim):
  model = RND(num_auxiliary_tasks=lap_dim)
  optim_mask = {'params': {'predictions': True, 'targets': False}}
  optim = optax.masked(optim, optim_mask)

  variables = model.init(rng, dummy_obs)
  params = variables['params']
  # batch_stats = variables['batch_stats']
  # We have to unfreeze the parameters or else optax has some issues
  # with optax.masked
  params = params.unfreeze()

  rnd_state = RNDState.create(
    apply_fn=model.apply,
    params={'params': params},
    optim=optim,
    # batch_stats=batch_stats
    )
  return rnd_state


def rnd_train_step(rnd_state, batch):

  def loss_fn(rnd_params):
    rnd_outputs = rnd_state.apply_fn(
        {**rnd_params},
        #  **{'batch_stats': rnd_state.batch_stats}},
        batch,
        # mutable=['batch_stats'],
        )
    loss = rnd_outputs.prediction_error.mean()
    return loss

  rnd_grads = jax.grad(loss_fn)(rnd_state.params)

  rnd_state = rnd_state.apply_gradients(grads=rnd_grads)
  # rnd_state = rnd_state.replace(batch_stats=updates['batch_stats'])

  return rnd_state


def get_rnd(rnd_state, rnd_params, obs):
  return rnd_state.apply_fn(
      {**rnd_params},
      #  , **{'batch_stats': rnd_state.batch_stats}},
      obs,
      )
