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
"""A DQN agent training on Atari.

From the paper "Human Level Control Through Deep Reinforcement Learning"
http://www.nature.com/articles/nature14236.
"""

# pylint: disable=g-bad-import-order

import collections
import itertools
import os
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import chex
import dm_env
import haiku as hk
import jax
from jax.config import config
import gym
import mon_minigrid
from mon_minigrid.custom_wrappers import dceo_wrapper
import numpy as np
import optax

from dqn_zoo import env_wrapper
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.dqn import plot_minig

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'MonMiniGrid-NineRooms-v0', '')
flags.DEFINE_string('algo', 'dceo', '')
flags.DEFINE_integer('replay_capacity', int(1e6), '')
flags.DEFINE_bool('compress_state', False, '')
flags.DEFINE_bool('plot', False, '')
flags.DEFINE_bool('reward_free', False, '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.001, '')
flags.DEFINE_integer('learn_period', 4, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('exploration_epsilon_begin_value', 1., '')
flags.DEFINE_float('exploration_epsilon_end_value', 0.1, '')
flags.DEFINE_float('exploration_epsilon_decay_frame_fraction', 0.1, '')
flags.DEFINE_float('eval_exploration_epsilon', 0.0, '')
flags.DEFINE_integer('target_network_update_period', int(1e3), '')
flags.DEFINE_float('grad_error_bound', 1. / 32, '')
flags.DEFINE_float('learning_rate', 0.0001, '')
flags.DEFINE_float('additional_discount', 0.99, '')
flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
flags.DEFINE_integer('n_steps', 3, '')
flags.DEFINE_integer('num_iterations', 1_000, '')
flags.DEFINE_integer('num_train_frames', int(1e3), '')  # Per iteration.
flags.DEFINE_integer('num_eval_frames', int(0e0), '')  # Per iteration.
flags.DEFINE_integer('lap_dim', 20, '')
flags.DEFINE_integer('num_options', 10, '')
flags.DEFINE_integer('avg_option_dur', 10, '')
flags.DEFINE_integer('option_learning_steps', 0, '')
flags.DEFINE_float('option_prob', 0.9, '')
flags.DEFINE_float('rnd_w', 0., '')
flags.DEFINE_string('results_csv_path', '/tmp/results/results.csv', '')
flags.DEFINE_string('plot_path', 'plots/monmini_cleaner/', '')


def main(argv):
  """Trains DQN agent on MonMinigrid."""
  del argv

  if FLAGS.algo == 'dceo':
    from dqn_zoo.dqn import dceo_agent as agent
  elif FLAGS.algo == 'rnd':
    from dqn_zoo.dqn import rnd_agent as agent
  else:
    raise NotImplementedError

  logging.info('DQN on MonMinigrid on %s.',
     jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64))

  if FLAGS.results_csv_path:
    writer = parts.CsvWriter(FLAGS.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder(reward_free):
    """Creates MonMiniGrid environment."""
    # import pdb;pdb.set_trace()
    gym_env = gym.make(FLAGS.environment_name)
    gym_env.set_reward_free(reward_free)
    gym_env = dceo_wrapper.DCEOWrapper(gym_env)
    return env_wrapper.GymWrapper(gym_env)

  env = environment_builder(FLAGS.reward_free)
  num_actions = env.action_spec().num_values
  network_fn = networks.minigrid_network(num_actions)
  action_network = hk.transform(network_fn)
  network_fn = networks.minigrid_network(FLAGS.lap_dim)
  lap_network = hk.transform(network_fn)

  def preprocessor_builder():
    return processors.minigrid(additional_discount=FLAGS.additional_discount)

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)

  sample_network_input = sample_processed_timestep.observation
  logging.info(f"Observation shape: {sample_network_input.shape}")

  FLAGS.option_learning_steps = (
      FLAGS.option_learning_steps * int(FLAGS.num_options > 0))
  FLAGS.exploration_epsilon_end_value = (
      FLAGS.exploration_epsilon_end_value if not FLAGS.reward_free else
      FLAGS.exploration_epsilon_begin_value)

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=FLAGS.option_learning_steps,
      decay_steps=int(FLAGS.exploration_epsilon_decay_frame_fraction *
                      FLAGS.num_iterations * FLAGS.num_train_frames),
      begin_value=FLAGS.exploration_epsilon_begin_value,
      end_value=FLAGS.exploration_epsilon_end_value)

  if FLAGS.compress_state:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t))

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t))
  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  transition_accumulator = replay_lib.NStepTransitionAccumulator(FLAGS.n_steps)
  replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, replay_structure,
                                       random_state, encoder, decoder)
  lap_transition_accumulator = replay_lib.TransitionAccumulator()
  lap_replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, 
                                       replay_structure,
                                       random_state, encoder, decoder)

  optimizer = optax.adam(learning_rate=FLAGS.learning_rate,)
  rep_optimizer = optax.adam(learning_rate=FLAGS.learning_rate,)

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent.Agent(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=action_network,
      lap_network=lap_network,
      optimizer=optimizer,
      rep_optimizer=rep_optimizer,
      transition_accumulator=transition_accumulator,
      lap_transition_accumulator=lap_transition_accumulator,
      replay=replay,
      lap_replay=lap_replay,
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
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  if FLAGS.plot:
    plotter = plot_minig.Plot(
        env, sample_processed_timestep, directory=FLAGS.plot_path)
  agent_pos_track = None

  while state.iteration <= FLAGS.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    # Reset the buffer in order to avoid benefits from pretraining data
    pretraining_ended_now = (FLAGS.option_learning_steps > 0 and 
        state.iteration * FLAGS.num_train_frames == FLAGS.option_learning_steps)
    if pretraining_ended_now:
      logging.info("Reseting buffer")
      transition_accumulator = replay_lib.NStepTransitionAccumulator(
          FLAGS.n_steps)
      replay = replay_lib.TransitionReplay(
          FLAGS.replay_capacity, replay_structure,
          random_state, encoder, decoder)
      train_agent.new_replay(replay, transition_accumulator)

    reward_free = (FLAGS.reward_free or
        state.iteration * FLAGS.num_train_frames < FLAGS.option_learning_steps)
    env = environment_builder(reward_free=reward_free)

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env)
    train_seq_truncated = itertools.islice(train_seq, FLAGS.num_train_frames)
    train_trackers = parts.make_monmini_trackers(train_agent, agent_pos_track)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)
    agent_pos_track = train_trackers[0]

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env)
    eval_seq_truncated = itertools.islice(eval_seq, FLAGS.num_eval_frames)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    # Logging and checkpointing.
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * FLAGS.num_train_frames, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_state_value', train_stats['state_value'], '%.3f'),
    ]
    log_output_str = '\n '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

    if FLAGS.plot:
      rep = train_agent.get_rep(plotter.cover)

      plotter.plot(
        rep,
        state.iteration, 
        min(min(10, FLAGS.lap_dim), len(rep[0])),
        name=f'rep'
        )

      # option_actions = []
      # for o in range(train_agent._cur_tot_opt):
      #   option_actions.append(np.argmax(np.array(
      #     action_network.apply(
      #     train_agent.options[o].online_params,
      #     train_rng_key,
      #     plotter.cover).q_values),axis=1)
      #   )
      # plotter.plot_actions(
      #   option_actions,
      #   state.iteration,
      #   name=f'option_actions')

      # main_actions = [np.argmax(np.array(
      #   action_network.apply(
      #   train_agent.online_params,
      #   train_rng_key,
      #   plotter.cover).q_values),axis=1)]

      # plotter.plot_actions(
      #   main_actions,
      #   state.iteration,
      #   name=f'main_actions')

      plotter.plot_pos(
          agent_pos_track.counts,
          state.iteration,
          name='pos')

  writer.close()


if __name__ == '__main__':
  config.config_with_absl()
  app.run(main)
