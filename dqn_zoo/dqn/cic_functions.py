
import functools
import operator
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union

import chex
from flax import core
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dqn_zoo import networks


@chex.dataclass
class CICOutput:
  query: chex.Array
  key: chex.Array
  state: chex.Array
  next_state: chex.Array


class CIC(nn.Module):
  skill_dim: int = 64
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, obs, next_obs, skill):
    obs = obs.astype(self.input_dtype)
    next_obs = next_obs.astype(self.input_dtype)
    skill = skill.astype(self.input_dtype)

    state = networks.PartiallyNatureDQNNetwork(
        self.skill_dim, name='state')(obs)

    next_state = networks.PartiallyNatureDQNNetwork(
        self.skill_dim, name='next_state')(next_obs)

    concat_states = jnp.concatenate([state, next_state], axis=1)

    skill_prediction = networks.LinearNet(
        self.skill_dim, name='prediction')(concat_states)

    skill_projection = networks.LinearNet(
        self.skill_dim, name='projection')(skill)

    return CICOutput(
        query=skill_projection,
        key=skill_prediction,
        state=state,
        next_state=next_state)


class CICState(struct.PyTreeNode):
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
             optim, 
             **kwargs):
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params),
        )


def create_cic_state(dummy_obs, rng, optim, lap_dim):
  model = CIC(skill_dim=lap_dim)

  params = model.init(
      rng, dummy_obs, dummy_obs, np.random.rand(lap_dim)[None,:])

  cic_state = CICState.create(
    apply_fn=model.apply,
    params=params,
    optim=optim,
    )
  return cic_state


def cic_train_step(cic_state, obs, next_obs, skill):
  temperature = 0.5
  eps = 1e-6

  def loss_fn(cic_params):
    cic_outputs = cic_state.apply_fn(
        cic_params, obs, next_obs, skill)
    query = cic_outputs.query
    key = cic_outputs.key
    query = query / jnp.maximum(
        jnp.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    key = key / jnp.maximum(
        jnp.linalg.norm(key, axis=1, keepdims=True), 1e-12)
    cov = jnp.matmul(query, key.T)  # (b, b)
    sim = jnp.exp(cov / temperature)
    neg = sim.sum(axis=-1)  # (b,)
    row_sub = jnp.power(jnp.e, 1 / temperature) * jnp.ones_like(neg)
    neg = jnp.clip(neg - row_sub, a_min=eps)  # clamp for numerical stability
    pos = jnp.exp(jnp.sum(query * key, axis=-1) / temperature)  # (b,)
    loss = -jnp.log(pos / (neg + eps))  # (b,)
    return loss.mean()

  cic_grads = jax.grad(loss_fn)(cic_state.params)

  cic_state = cic_state.apply_gradients(grads=cic_grads)

  return cic_state


def get_cic(cic_state, cic_params, next_obs, skill):
  return cic_state.apply_fn(cic_params, next_obs, next_obs, skill)
