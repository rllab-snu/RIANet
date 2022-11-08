#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla

import copy
import glob
import os
import time
import datetime
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from carla_env import CarlaVirtualEnv

class RL_Trainer(object):
    def __init__(self, args):
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.args = args

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.cuda and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        torch.set_num_threads(1)
        self.device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

        self.envs = make_vec_envs('carla-env', args.seed, args.num_processes,
                                  args.gamma, args.log_dir, self.device, False, args=args)

        args.observation_length = self.envs.venv.envs[0].env.observation_length
        args.act_dim = self.envs.action_space  # 10 # 5
        #args.space_shape = self.envs.venv.envs[0].env.space_shape
        #args.space_len = self.envs.venv.envs[0].env.space_len

        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={'recurrent': False},
            args=args)
        self.actor_critic.to(self.device)

        if args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)

        self.rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                       self.envs.observation_space.shape, self.envs.action_space,
                                       self.actor_critic.recurrent_hidden_state_size)
        self.episode_rewards = deque(maxlen=10)
        self.episode_success = deque(maxlen=10)
        #self.episode_v = deque(maxlen=10)
        self.episode_rewards2 = deque(maxlen=100)
        self.episode_success2 = deque(maxlen=100)
        #self.episode_v2 = deque(maxlen=100)

        self.total_rewards = []
        self.total_success = []
        self.total_episodes = 0

        self.num_updates = int(
            args.num_env_steps) // args.num_steps // args.num_processes

        self.current_update_time = 0
        self.current_step = 0

        self.reset_started = False
        self.start_time = time.time()

    def reset_scenario_and_state(self):
        self.envs.venv.envs[0].env.reset_scenario_and_state()

    def update_virtual_env_and_get_control(self, env_data):
        self.envs.venv.envs[0].env.update_state(env_data)

        if self.current_step == 0:
            if self.args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, self.current_update_time, self.num_updates,
                    self.agent.optimizer.lr if self.args.algo == "acktr" else self.args.lr)

            sys.stdout.write(
                '\r' + ' '*40 + '-' * 5 + ' {} / {} '.format(self.current_update_time % self.args.log_interval,
                                                self.args.log_interval) + '-' * 5 + ' collecting data...' + ' '*10);
            sys.stdout.flush()

        if not self.reset_started:
            obs = self.envs.reset()

            self.rollouts.obs[0].copy_(obs)
            self.rollouts.to(self.device)
        else:
            obs, reward, done, infos = self.envs.step(self.action)

            for info in infos:
                if 'episode' in info.keys():
                    self.total_rewards.append(info['episode']['r'])
                    self.total_success.append(info['is_success'])
                    self.total_episodes += 1

                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_success.append(info['is_success'])
                    #self.episode_v.append(info['v'])
                    self.episode_rewards2.append(info['episode']['r'])
                    self.episode_success2.append(info['is_success'])
                    #self.episode_v2.append(info['v'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            self.rollouts.insert(obs, self.recurrent_hidden_states, self.action,
                                 self.action_log_prob, self.value, reward, masks, bad_masks)

            # update network
            if self.current_step == self.args.num_steps - 1:
                sys.stdout.write(
                    '\r' + ' ' * 40 + '-' * 5 + ' {} / {} '.format(self.current_update_time % self.args.log_interval,
                                                                    self.args.log_interval) + '-' * 5 + ' training...' + ' ' * 10);
                sys.stdout.flush()

                with torch.no_grad():
                    next_value = self.actor_critic.get_value(
                        self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1]).detach()

                self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                              self.args.gae_lambda, self.args.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
                self.rollouts.after_update()

                # save for every interval-th episode or for the last epoch
                if (
                        self.current_update_time % self.args.save_interval == 0 or self.current_update_time == self.num_updates - 1) \
                        and self.args.save_dir != "":
                    save_path = os.path.join(self.args.save_dir, self.args.algo)
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    torch.save([
                        self.actor_critic,
                        getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
                    ], os.path.join(save_path,
                                    "{}_{}_seed{}_{}.pt".format(self.args.env_name, self.args.net_model_type, \
                                                                self.args.seed, str(self.current_update_time))))

                # print result
                if (self.current_update_time % self.args.log_interval == 0 and len(self.episode_rewards) > 1):
                    total_num_steps = (self.current_update_time + 1) * self.args.num_processes * self.args.num_steps
                    end_time = time.time()
                    print("Updates {}, num timesteps {}, num episodes {}, FPS {}, time elapsed {}"
                          .format(self.current_update_time, total_num_steps, self.total_episodes,
                                  int(total_num_steps / (end_time - self.start_time)),
                                  str(datetime.timedelta(seconds=int(end_time - self.start_time)))))
                    print(
                        "Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}"
                            .format(len(self.episode_rewards), np.mean(self.episode_rewards),
                                    np.median(self.episode_rewards), np.min(self.episode_rewards),
                                    np.max(self.episode_rewards)))
                    print(
                        "Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}"
                            .format(len(self.episode_rewards2), np.mean(self.episode_rewards2),
                                    np.median(self.episode_rewards2), np.min(self.episode_rewards2),
                                    np.max(self.episode_rewards2)))
                    print("")

        # Sample actions
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[self.current_step], self.rollouts.recurrent_hidden_states[self.current_step],
                self.rollouts.masks[self.current_step])

        if not self.reset_started:
            self.reset_started = True
        elif self.current_step == self.args.num_steps - 1:
            self.current_step = 0
            self.current_update_time += 1
        else:
            self.current_step += 1

        return self.envs.venv.envs[0].env.translate_action(self.action.cpu().numpy().reshape(-1))

class TrainScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, args):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.args = args

        self.RL_Trainer = RL_Trainer(args)

        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = self.args.debug > 1
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(self.args.timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent._agent.RL_Trainer = self.RL_Trainer  # attach RL trainer
        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        ResultOutputProvider(self, global_result)
