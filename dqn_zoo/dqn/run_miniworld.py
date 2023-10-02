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
"""A DQN agent training on MiniWorld.

From the paper "Human Level Control Through Deep Reinforcement Learning"
http://www.nature.com/articles/nature14236.
"""

# pylint: disable=g-bad-import-order

import collections
import itertools
import os
import sys
import typing
import pickle

from absl import app
from absl import flags
from absl import logging
import chex
import dm_env
import haiku as hk
import jax
from jax.config import config
import numpy as np
import optax
import gym
import gym_miniworld

from dqn_zoo import env_wrapper
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.dqn import plot_miniw
from dqn_zoo.dqn import plot_miniw_actions
from dqn_zoo.dqn import plot_mwh

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'MiniWorld-FourRooms-v0', '')
flags.DEFINE_string('algo', 'dceo', '')
flags.DEFINE_integer('replay_capacity', int(1e6), '')
flags.DEFINE_bool('compress_state', True, '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.01, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('exploration_epsilon_begin_value', 1., '')
flags.DEFINE_float('exploration_epsilon_end_value', 0.1, '')
flags.DEFINE_float('exploration_epsilon_decay_frame_fraction', 0.1, '')
flags.DEFINE_float('eval_exploration_epsilon', 0.05, '')
flags.DEFINE_integer('target_network_update_period', int(5e3), '')
flags.DEFINE_float('grad_error_bound', 1. / 32, '')
flags.DEFINE_float('learning_rate', 0.00025, '')
flags.DEFINE_float('optimizer_epsilon', 0.01 / 32**2, '')
flags.DEFINE_float('additional_discount', 0.99, '')
flags.DEFINE_bool('plot', False, '')
flags.DEFINE_bool('reward_free', False, '')
flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
flags.DEFINE_integer('n_steps', 5, '')
flags.DEFINE_integer('num_iterations', 200, '')
flags.DEFINE_integer('num_train_frames', int(1e3), '')  # Per iteration.
flags.DEFINE_integer('num_eval_frames', int(0e0), '')  # Per iteration.
flags.DEFINE_integer('lap_dim', 512, '')
flags.DEFINE_integer('num_options', 0, '')
flags.DEFINE_integer('avg_option_dur', 10, '')
flags.DEFINE_integer('option_learning_steps', 0, '')
flags.DEFINE_float('option_prob', 0., '')
flags.DEFINE_float('rnd_w', 0., '')
flags.DEFINE_integer('learn_period', 4, '')
flags.DEFINE_string('results_csv_path', '/tmp/results.csv', '')
flags.DEFINE_string('plot_path', 'plots/miniw/', '')


def main(argv):
  """Trains DQN agent on MiniWorld."""
  del argv

  if FLAGS.algo == 'dceo':
    from dqn_zoo.dqn import dceo_agent as agent
  elif FLAGS.algo == 'cic':
    from dqn_zoo.dqn import cic_agent as agent
  elif FLAGS.algo == 'rnd':
    from dqn_zoo.dqn import rnd_agent as agent
  else:
    raise NotImplementedError

  logging.info('Collecting data on MiniWorld on %s.',
     jax.lib.xla_bridge.get_backend().platform)
  np.random.seed(FLAGS.seed)
  random_state = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64))

  if FLAGS.results_csv_path:
    writer = parts.CsvWriter(FLAGS.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates MiniWorld environment."""
    gym_env = gym.make(FLAGS.environment_name)
    gym_env.seed(FLAGS.seed)
    return env_wrapper.GymWrapper(gym_env)

  env = environment_builder()
  num_actions = env.action_spec().num_values
  network_fn = networks.dqn_atari_network(FLAGS.lap_dim)
  lap_network = hk.transform(network_fn)
  network_fn = networks.dqn_atari_network(num_actions)
  action_network = hk.transform(network_fn)

  def preprocessor_builder():
    return processors.miniworld(
        additional_discount=FLAGS.additional_discount,
        resize_shape=None,
        grayscaling=False,)

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)
  sample_network_input = sample_processed_timestep.observation
  logging.info(f"Observation shape: {sample_network_input.shape}")

  FLAGS.num_options = (FLAGS.num_options * int(FLAGS.option_prob > 0.))
  FLAGS.option_learning_steps = (
      FLAGS.option_learning_steps * int(FLAGS.num_options > 0))
  FLAGS.exploration_epsilon_end_value = (
      FLAGS.exploration_epsilon_end_value if not FLAGS.reward_free else
      FLAGS.exploration_epsilon_begin_value)

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity),
      decay_steps=int(FLAGS.exploration_epsilon_decay_frame_fraction *
                      FLAGS.num_iterations * FLAGS.num_train_frames),
      begin_value=FLAGS.exploration_epsilon_begin_value,
      end_value=FLAGS.exploration_epsilon_end_value)

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  transition_accumulator = replay_lib.NStepTransitionAccumulator(FLAGS.n_steps)
  replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, replay_structure,
                                       random_state)

  optimizer = optax.rmsprop(
      learning_rate=FLAGS.learning_rate,
      decay=0.95,
      eps=FLAGS.optimizer_epsilon,
      centered=True,
  )

  rep_optimizer = optax.adam(
      learning_rate=0.0001,
  )

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent.Agent(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=action_network,
      lap_network=lap_network,
      optimizer=optimizer,
      rep_optimizer=rep_optimizer,
      transition_accumulator=transition_accumulator,
      lap_transition_accumulator=transition_accumulator,
      replay=replay,
      lap_replay=replay,
      batch_size=FLAGS.batch_size,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
      learn_period=FLAGS.learn_period,
      target_network_update_period=FLAGS.target_network_update_period,
      grad_error_bound=FLAGS.grad_error_bound,
      rng_key=train_rng_key,
      num_actions=num_actions,
      num_options=FLAGS.num_options,
      lap_dim=FLAGS.lap_dim,
      option_prob=FLAGS.option_prob,
      avg_option_dur=FLAGS.avg_option_dur,
      option_learning_steps=FLAGS.option_learning_steps,
      reward_free=FLAGS.reward_free,
      learning_rate=FLAGS.learning_rate,
      rnd_w=FLAGS.rnd_w,
  )

  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=action_network,
      exploration_epsilon=FLAGS.eval_exploration_epsilon,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  if FLAGS.plot:
    if 'MyWayHome' in FLAGS.environment_name:
      plotter = plot_mwh.Plot(
        'datasets/mwh_plotting_1st_person.pkl', FLAGS.plot_path)
    else:
      plotter = plot_miniw_actions.Plot(
        # 'datasets/4r_textures_plotting_clean.pkl', FLAGS.plot_path)
        'datasets/4r_actions_1st_person.pkl', FLAGS.plot_path)
  agent_pos_track = None
  eval_pos_track = None

  while state.iteration <= FLAGS.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    logging.info('Data collection iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env)
    train_seq_truncated = itertools.islice(train_seq, FLAGS.num_train_frames)
    train_trackers = parts.make_miniw_trackers(train_agent, agent_pos_track)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)
    agent_pos_track = train_trackers[0]

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env)
    eval_seq_truncated = itertools.islice(eval_seq, FLAGS.num_eval_frames)
    eval_trackers = parts.make_miniw_trackers(eval_agent, eval_pos_track)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    # Logging and checkpointing.
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * FLAGS.num_train_frames, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
    ]
    log_output_str = '\n '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

    if FLAGS.plot:
      if 'Actions' in FLAGS.environment_name:
        orient_dict = {0: 'front',
                       1: 'left',
                       2: 'back',
                       3: 'right',}
        for orientation in range(4):
          rep = train_agent.get_rep(plotter.cover[orientation])
          plotter.plot(
              rep,
              state.iteration,
              min(min(10, FLAGS.lap_dim),
              len(rep[0])),
              name=f'rep_{orient_dict[orientation]}')
      else:
        rep = train_agent.get_rep(plotter.cover[orientation])
        plotter.plot(
            rep,
            state.iteration,
            min(min(10, FLAGS.lap_dim),
            len(rep[0])),
            name='rep')

      plotter.plot_pos(
          [agent_pos_track.bin_counts],
          state.iteration,
          name='pos')

if __name__ == '__main__':
  config.config_with_absl()
  app.run(main)
