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
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class Option:
  def __init__(self, online_params, target_params, opt_state):
    self.online_params = online_params
    self.target_params = target_params
    self.opt_state = opt_state


class Agent(parts.Agent):
  """Deep Covering Options agent."""

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
      avg_option_dur: int,
      option_learning_steps: int,
      lap_dim: int,
      option_prob: float,
      reward_free: bool,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._lap_replay = lap_replay
    self._transition_accumulator = transition_accumulator
    self._lap_transition_accumulator = lap_transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period
    self._num_actions = num_actions
    self._num_options = num_options
    self._option_prob = option_prob * int(num_options > 0)
    self._d = avg_option_dur * int(num_options > 0)
    self._option_learning_steps = option_learning_steps * int(num_options > 0)
    self._reward_free = reward_free

    # Initialize network parameters and optimizer.
    rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input['state'][None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    self.options = []
    for o in range(num_options):
      rng_key, network_rng_key = jax.random.split(rng_key)
      online_params = network.init(network_rng_key,
                                      sample_network_input['state'][None, ...])
      target_params = online_params
      opt_state = optimizer.init(self._online_params)
      self.options.append(Option(
          online_params=online_params,
          target_params=target_params,
          opt_state=opt_state))

    self._laplacian_params = []
    self._lap_opt_state = []
    for o in range(num_options):
      rng_key, network_rng_key = jax.random.split(rng_key)
      self._laplacian_params.append(lap_network.init(network_rng_key,
                                        sample_network_input['obs'][None, ...]))
      self._lap_opt_state.append(rep_optimizer.init(self._laplacian_params[-1]))
    self._rng_key = rng_key

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}
    self._option_term = True
    self._cur_opt = None
    self._cur_tot_opt = 1
    self._option_period = num_options and option_learning_steps // num_options

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      _, *apply_keys = jax.random.split(rng_key, 4)
      q_tm1 = network.apply(online_params, apply_keys[0],
                            transitions.index_tm1).q_values
      q_t = network.apply(online_params, apply_keys[1],
                          transitions.index_t).q_values
      q_target_t = network.apply(target_params, apply_keys[2],
                                 transitions.index_t).q_values
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
      grads = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(grads, opt_state)
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

    coeff_vector = jnp.arange(lap_dim, 0, -1)
    coeff_vector = np.concatenate((coeff_vector, np.zeros(1)))

    def neg_loss_fn(phi_u, phi_v):
      loss = 0
      for dim in range(lap_dim, 0, -1):
        coeff = coeff_vector[dim-1] - coeff_vector[dim]
        x_norm = jnp.sqrt(jnp.dot(phi_u[:dim], phi_u[:dim]))
        y_norm = jnp.sqrt(jnp.dot(phi_v[:dim], phi_v[:dim]))
        dot_product = jnp.dot(phi_u[:dim], phi_v[:dim])
        loss += coeff * (
          dot_product ** 2 - x_norm ** 2 / lap_dim  - y_norm ** 2 / lap_dim)
      return loss

    neg_loss_vmap = jax.vmap(neg_loss_fn)

    def _update_lap(
      rng_key, opt_state, params, transitions, transitions_u, transitions_v):
      """Computes learning update from batch of replay transitions."""
      rng_key, *update_keys = jax.random.split(rng_key, 5)

      def lap_loss_fn(params, update_keys):
        """Calculates loss given network parameters and transitions."""
        phi_tm1 = lap_network.apply(params, update_keys[0],
                              transitions.s_tm1).q_values
        phi_t = lap_network.apply(params, update_keys[1],
                                   transitions.s_t).q_values
        phi_u = lap_network.apply(params, update_keys[2],
                                   transitions_u.s_tm1).q_values
        phi_v = lap_network.apply(params, update_keys[3],
                                   transitions_v.s_tm1).q_values
        pos_loss = ((phi_tm1 - phi_t)**2).dot(coeff_vector[:lap_dim])
        neg_loss  = neg_loss_vmap(phi_u, phi_v)
        loss = pos_loss + 2. * neg_loss
        chex.assert_shape(loss, (self._batch_size,))
        loss = jnp.mean(loss)
        return loss, (jnp.mean(pos_loss), jnp.mean(neg_loss))

      grads, (pos_loss, neg_loss) = jax.grad(
        lap_loss_fn, has_aux=True)(params, update_keys)
      updates, new_opt_state = rep_optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return rng_key, new_opt_state, new_params, pos_loss, neg_loss

    self.update_lap = jax.jit(_update_lap)

    def _get_lap(rng_key, network_params, obs):
      rng_key, apply_key = jax.random.split(rng_key, 2)
      return rng_key, lap_network.apply(network_params, apply_key, obs).q_values

    self.get_lap = jax.jit(_get_lap)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    if self._preprocessor is not None:
      timestep = self._preprocessor(timestep)

    self._option_term = self._option_term or np.random.rand() < (1 / self._d)

    if self._replay.size < self._min_replay_capacity:
      option_prob = 0.
    else:
      option_prob = self._option_prob

    if self._option_term:
      self._cur_opt = None

      if np.random.rand() < self.exploration_epsilon:
        if np.random.rand() < option_prob:
          self._cur_opt = np.random.randint(self._cur_tot_opt)
          self._option_term = False
          action = self._action = self._act(timestep, self._cur_opt)
        else:
          action = self._action = np.random.randint(self._num_actions)
      else:
        action = self._action = self._act(timestep, self._cur_opt)

    else:
      assert self._cur_opt is not None
      action = self._action = self._act(timestep, self._cur_opt)

    for transition in self._transition_accumulator.step(timestep, action):
      self._replay.add(transition)
    for transition in self._lap_transition_accumulator.step(timestep, action):
      self._lap_replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action
    
    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._option_period > 0 and (self._frame_t-1) % self._option_period == 0:
      if self._cur_tot_opt < self._num_options:
        logging.info("Adding a new option")
        self._cur_tot_opt = self._cur_tot_opt + 1

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params
      for option in self.options:
        option.target_params = option.online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    self._lap_transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep, o) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation['state']
    online_params = (self.options[o].online_params if self._cur_opt is not None
        else self._online_params)
    self._rng_key, a_t, v_t = self.select_action(self._rng_key,
                                                  online_params, s_t, 0.)
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    logging.log_first_n(logging.INFO, 'Begin learning', 1)

    # Update Laplacian representation
    if (
        self._num_options > 0 and (
        self._option_learning_steps == 0 or
        self._frame_t <= self._option_learning_steps) 
       ):

      transitions = self._lap_replay.sample(self._batch_size)
      transitions_u = self._lap_replay.sample(self._batch_size)
      transitions_v = self._lap_replay.sample(self._batch_size)
      (
          self._rng_key, 
          self._lap_opt_state[self._cur_tot_opt-1], 
          self._laplacian_params[self._cur_tot_opt-1], 
          pos_loss,
          neg_loss
      ) = self.update_lap(
          self._rng_key,
          self._lap_opt_state[self._cur_tot_opt-1],
          self._laplacian_params[self._cur_tot_opt-1],
          transitions,
          transitions_u,
          transitions_v
      )

      # Update option policies
      for o, option in enumerate(self.options):
        if o >= self._cur_tot_opt:
          continue
        transitions = self._replay.sample(self._batch_size)
        self._rng_key, lap_rep = self.get_lap(
            self._rng_key,
            self._laplacian_params[o],
            np.vstack((transitions.s_tm1, transitions.s_t))
            )
        lap_rep = np.array(lap_rep)
        intr_rew = lap_rep[self._batch_size:, 1] - lap_rep[:self._batch_size, 1]
        transitions = transitions._replace(r_t=intr_rew)

        (
            self._rng_key,
            option.opt_state,
            option.online_params
        ) = self.update(
            self._rng_key,
            option.opt_state,
            option.online_params,
            option.target_params,
            transitions,
        )

    # Update main learner
    if not self._reward_free and self._frame_t > self._option_learning_steps:
      transitions = self._replay.sample(self._batch_size)
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

  @property
  def laplacian_params(self) -> float:
    """Returns the current Laplacian parameters."""
    return self._laplacian_params[self._cur_tot_opt-1]

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
    reps = []
    for opt_idx in range(self._cur_tot_opt):
      self._rng_key, rep_key = jax.random.split(self._rng_key)
      (
          self._rng_key,
          rep,
      ) = self.get_lap(
          rep_key, 
          self._laplacian_params[opt_idx],
          cover
      )
      reps.append(np.array(rep)[:, 1])
    return np.array(reps).T
