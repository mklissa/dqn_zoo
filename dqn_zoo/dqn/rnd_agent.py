# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DQN agent class."""

# pylint: disable=g-bad-import-order

import time
from typing import Any, Callable, Mapping, Text

from absl import logging
import chex
import distrax
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.dqn.rnd_functions import rnd_train_step, create_rnd_state
from dqn_zoo.dqn.rnd_functions import get_rnd

# Batch variant of q_learning.
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class Agent(parts.Agent):
  """RND agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      lap_network: parts.Network,
      optimizer: optax.GradientTransformation,
      rep_optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      lap_transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      lap_replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      grad_error_bound: float,
      rng_key: parts.PRNGKey,
      num_actions: int,
      num_options: int,
      lap_dim: int,
      option_prob: float,
      avg_option_dur: int,
      option_learning_steps: int,
      reward_free: bool,
      learning_rate: float,
      rnd_w: float,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period
    self._num_actions = num_actions
    self._num_options = num_options
    self._lap_dim = lap_dim
    self._option_prob = option_prob * int(num_options > 0)
    self._d = avg_option_dur
    self._option_learning_steps = option_learning_steps * int(num_options > 0)
    self._reward_free = reward_free
    self._rnd_w = rnd_w

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}
    self._cur_opt = None

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      _, *apply_keys = jax.random.split(rng_key, 4)
      q_tm1 = network.apply(online_params, apply_keys[0],
                            transitions.s_tm1).q_values
      q_t = network.apply(online_params, apply_keys[1],
                          transitions.s_t).q_values
      q_target_t = network.apply(target_params, apply_keys[2],
                                 transitions.s_t).q_values
      td_errors = _batch_double_q_learning(
          q_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          q_target_t,
          q_t,
      )
      td_errors = rlax.clip_gradient(td_errors, -grad_error_bound,
                                     grad_error_bound)
      losses = rlax.l2_loss(td_errors)
      chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses)
      return loss

    def _update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    self.update = jax.jit(_update)

    def _select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = distrax.EpsilonGreedy(q_t,
                                  exploration_epsilon).sample(seed=policy_key)
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self.select_action = jax.jit(_select_action)

    self._rng_key, rnd_key = jax.random.split(self._rng_key)
    self.rnd_state = create_rnd_state(sample_network_input[None, ...],
                                      rnd_key, rep_optimizer, lap_dim)
    self.rnd_train_step = jax.jit(rnd_train_step)
    # self.rnd_train_step = rnd_train_step
    self.get_rnd = jax.jit(get_rnd)
    self.reward_rms = replay_lib.RMS()
    self.obs_rms = replay_lib.RMS(shape=sample_network_input[None, ...].shape)
    self.rnd_reward_var = 1.
    self.obs_mean = 0.
    self.obs_var = 1.

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    if self._preprocessor is not None:
      timestep = self._preprocessor(timestep)

    if np.random.rand() < self.exploration_epsilon:
      action = self._action = np.random.randint(self._num_actions)
    else:
      action = self._action = self._act(timestep)

    for transition in self._transition_accumulator.step(timestep, action):
      self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self.select_action(self._rng_key,
                                                 self._online_params, s_t, 
                                                 0.0)
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    logging.log_first_n(logging.INFO, 'Begin learning', 1)

    transitions = self._replay.sample(self._batch_size)

    if self._rnd_w > 0.0: 
      # Update RND and add intrinsic rewards
      self.obs_mean, self.obs_var = self.obs_rms(transitions.s_tm1)
      rnd_batch = ((transitions.s_tm1 - self.obs_mean)
                    / (np.sqrt(self.obs_var) + 1e-5)).clip(-5., 5.)
      self.rnd_state = self.rnd_train_step(self.rnd_state, rnd_batch)
      rnd_output = self.get_rnd(
          self.rnd_state,
          self.rnd_state.params,
          rnd_batch)
      prediction_error = np.array(rnd_output.prediction_error)
      _, self.rnd_reward_var = self.reward_rms(prediction_error[:, None])
      int_rewards = (prediction_error /
          (np.sqrt(self.rnd_reward_var) + 1e-8)).astype(float)
      # int_rewards = prediction_error
      rnd_rewards = self._rnd_w * int_rewards + transitions.r_t
      transitions = transitions._replace(r_t=rnd_rewards)

    # Update main learner
    (
        self._rng_key,
        self._opt_state,
        self._online_params
    ) = self.update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values())
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def new_replay(self, replay, transition_accumulator):
    self._replay = replay
    self._transition_accumulator = transition_accumulator

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])

  def get_rep(self, cover):
    cover = ((cover - self.obs_mean)
          / (np.sqrt(self.obs_var) + 1e-5)).clip(-5., 5.)
    rnd_output = self.get_rnd(
        self.rnd_state, self.rnd_state.params, cover)
    prediction_error = np.array(rnd_output.prediction_error)
    int_rewards = (prediction_error /
        (np.sqrt(self.rnd_reward_var) + 1e-8)).astype(float)
    # int_rewards = prediction_error
    rep = int_rewards[:, None]
    return rep
