import numpy as np

import yaml

import time

from PIL import Image

import os

from collections import deque

import random

from utils import make_env

from wrappers.self_play_wrapper import SelfPlayWrapper


class Model:

    def __init__(self, pred_func):
        self.pred_func = pred_func

    def predict(self, obs, deterministic=True):
        return self.pred_func(obs), None


class Agent:

    def __init__(self, model_path=None, algorithm=None, model=None, env=None):
        assert model_path is None or model is None

        if model_path:
            assert algorithm is not None

            self.load_model(model_path, algorithm)

        self.model = model
        self.env = env

    def _calc_manhattan_dist(self, pos1, pos2):
        dist = abs(pos1[0] - pos2[0])
        dist += abs(pos1[1] - pos2[1])

        return dist

    def _find_closest_empty_pit(self, board, current_pos):
        current_pos_to_pit_pos_dists = []
        pit_pos_list = []
        for pit_cell in (3, 5): # for all not-filled pits
            pit_pos_Is, pit_pos_Js = np.where(board == pit_cell)
            if len(pit_pos_Is) == 0:
                continue

            for pit_pos in zip(pit_pos_Is, pit_pos_Js):
                pit_pos_list.append(pit_pos)
                current_pos_to_pit_pos_dists.append(
                    self._calc_manhattan_dist(current_pos, pit_pos)
                )

        pit_pos_and_dists = list(zip(pit_pos_list, current_pos_to_pit_pos_dists))
        pit_pos_and_dists = sorted(pit_pos_and_dists, key=lambda x: x[1])

        if len(pit_pos_and_dists) > 0:
            closest_pit_pos = pit_pos_and_dists[0]
        else:
            closest_pit_pos = None, None

        return closest_pit_pos

    def _optimal_orb_choice_from_agent_to_pits_dist(self, board, player_pos):
        min_dists = []
        orb_pos_list = []
        for orb_cell in (2, 4): # for all orbs
            orb_pos_Is, orb_pos_Js = np.where(board == orb_cell)
            if len(orb_pos_Is) == 0:
                continue

            for orb_pos in zip(orb_pos_Is, orb_pos_Js):
                orb_pos_list.append(orb_pos)
                agent_to_orb_dist = self._calc_manhattan_dist(orb_pos, player_pos)
                _, min_orb_to_pit_dist = self._find_closest_empty_pit(board, orb_pos)

                if min_orb_to_pit_dist is None:
                    return None

                min_dists.append(agent_to_orb_dist+min_orb_to_pit_dist)

        orb_pos_and_min_dists = list(zip(orb_pos_list, min_dists))
        orb_pos_and_min_dists = sorted(orb_pos_and_min_dists, key=lambda x: x[1])

        if len(orb_pos_and_min_dists) > 0:
            optimal_orb_pos = orb_pos_and_min_dists[0][0]
        else:
            optimal_orb_pos = None

        return optimal_orb_pos

    def _calc_direction(self, destination_pos, player_pos):
        i_diff = player_pos[0] - destination_pos[0]
        j_diff = player_pos[1] - destination_pos[1]

        if abs(i_diff) > abs(j_diff):
            if i_diff < 0:
                return 3
            elif i_diff > 0:
                return 1
        else:
            if j_diff < 0:
                return 2
            elif j_diff > 0:
                return 0

    def _change_direction_to(self, target_direction, player_direction):
        if player_direction == target_direction:
            return 1

        return 0

    def _explore_board(self, board, player_pos, player_direction, do_random=False):
        board_shape = board.shape

        board_center = board_shape[0]//2, board_shape[1]//2
        board_corner0 = 0, 0
        board_corner1 = 0, board_shape[1]-1
        board_corner2 = board_shape[0]-1, 0
        board_corner3 = board_shape[0]-1, board_shape[1]-1

        proposed_destinations = (board_center, board_corner0, 
                                board_corner1, board_corner2, 
                                board_corner3)
        if not do_random:
            dists = []
            for proposed_destination in proposed_destinations:
                distance = self._calc_manhattan_dist(player_pos, proposed_destination)
                dists.append(distance)

            destination_pos_and_dists = list(zip(proposed_destinations, dists))
            destination_pos_and_dists = sorted(destination_pos_and_dists, key=lambda x: x[1])
            max_dist_destination_pos = destination_pos_and_dists[-1][0]
        else:
            max_dist_destination_pos = random.choice(proposed_destinations)

        target_direction = self._calc_direction(max_dist_destination_pos, player_pos)
        action = self._change_direction_to(target_direction, player_direction)
        return action

    def _find_other_player_pos(self, board, player_pos):
        for player_cell in (1, 4, 5, 7): # for all players
            player_pos_Is, player_pos_Js = np.where(board == player_cell)
            if len(player_pos_Is) == 0:
                continue

            for other_player_pos in zip(player_pos_Is, player_pos_Js):
                if other_player_pos[0] != player_pos[0] or other_player_pos[1] != player_pos[1]:
                    return other_player_pos

        return None

    def _check_players_collision(self, board, player_pos):
        other_player_pos = self._find_other_player_pos(board, player_pos)
        if other_player_pos is not None:
            dist = self._calc_manhattan_dist(other_player_pos, player_pos)
        else:
            return False

        if dist in (1, 2):
            return True

        return False

    def _find_closest_other_filled_pit(self, board, player_pos, filled_pits_positions):
        other_pit_pos_list = []
        dists = []

        pit_pos_Is, pit_pos_Js = np.where(board == 7)
        if len(pit_pos_Is) == 0:
            return None, None

        for pit_pos in zip(pit_pos_Is, pit_pos_Js):
            if pit_pos not in filled_pits_positions:
                other_pit_pos_list.append(pit_pos)
                dists.append(self._calc_manhattan_dist(player_pos, pit_pos))

        pit_pos_and_dists = list(zip(other_pit_pos_list, dists))
        pit_pos_and_dists = sorted(pit_pos_and_dists, key=lambda x: x[1])

        if len(pit_pos_and_dists) > 0:
            closest_pit_pos = pit_pos_and_dists[0]
        else:
            closest_pit_pos = None, None

        return closest_pit_pos

    def _1v1_rules_function(self, obs):
        board = obs["board"]
        player_direction = obs["player0_direction"]
        player_has_orb = obs["player0_has_orb"]
        player_pos = obs["player0_position"]
        filled_pits_positions = obs["filled_pits_positions"]

        if self._check_players_collision(board, player_pos):
            action = self._explore_board(board, player_pos, player_direction, do_random=True)
            return action

        closest_other_filled_pit_pos, dist_to_closest_other_filled_pit = self._find_closest_other_filled_pit(board, player_pos, filled_pits_positions)
        if closest_other_filled_pit_pos is not None and dist_to_closest_other_filled_pit < 14:
            target_direction = self._calc_direction(closest_other_filled_pit_pos, player_pos)
            action = self._change_direction_to(target_direction, player_direction)
            return action

        if board[player_pos[0], player_pos[1]] == 7:
            if player_pos not in filled_pits_positions:
                return 4

        if player_has_orb:
            closest_pit_pos, _ = self._find_closest_empty_pit(board, player_pos)
            if closest_pit_pos is None:
                action = self._explore_board(board, player_pos, player_direction)
                return action

            if closest_pit_pos[0] == player_pos[0] and closest_pit_pos[1] == player_pos[1]:
                return 3
            
            target_direction = self._calc_direction(closest_pit_pos, player_pos)
            action = self._change_direction_to(target_direction, player_direction)
            return action

        optimal_orb_pos = self._optimal_orb_choice_from_agent_to_pits_dist(board, player_pos)
        if optimal_orb_pos is None:
            action = self._explore_board(board, player_pos, player_direction)
            return action
        
        if optimal_orb_pos[0] == player_pos[0] and optimal_orb_pos[1] == player_pos[1]:
            return 2
        
        target_direction = self._calc_direction(optimal_orb_pos, player_pos)
        action = self._change_direction_to(target_direction, player_direction)
        return action

    def run_rules_one_step(self, obs, rule_type):
        match rule_type:
            case "1v1":
                return self._1v1_rules_function(obs)
            case _:
                raise NotImplemented

    def convert_rules_to_model(self, rule_type):
        match rule_type:
            case "1v1":
                self.model = Model(self._1v1_rules_function)
            case _:
                raise NotImplemented

    def run_agent_one_episode(self, env, max_steps=500, fps=None, 
                deterministic_action_choice=False, 
                print_rewards=True, return_frames=True):
        assert return_frames and env.game._pygame_mode

        obs = env.reset()
        done = False

        rewards = []
        frames = []
        start = time.time()

        for step in range(1, max_steps+1):
            action = self.predict(obs, deterministic=deterministic_action_choice)

            obs, reward, done, info = env.step(action)

            rewards.append(reward)

            if print_rewards:
                print(f"\rTotal Rewards: {sum(rewards):.4f}, Current Reward: {reward:.4f}, Taken Action: {action}", end="")

            if return_frames:
                frame = env.render("rgb_array")
                frames.append(frame)

            if done:
                print()
                print("One episode is done with ", end="")
                for i in range(env.game.team_num):
                    for j in range(env.game.team_size):
                        print(f"(team{i}_player{j} having done {info[f'team{i}_player{j}_movements#']} movements) ", end="")
                print()
                break

            if fps is not None:
                env.game.clock.tick(fps)

        print()

        if not print_rewards:
            print("Total Rewards:", sum(rewards))

        print("Final Info:", info)
        print("Episode Length:", step+1)
        print()
        print("Episode FPS:", int(step/(time.time()-start)))

        return rewards, frames

    def train_with_self_play(self, env, clone_model, models_bank_maxlen=5, 
                            self_play_epochs=20, iterations=5_000_000, 
                            log_interval=1_000, models_bank_path="./models/self-play", 
                            algorithm="A2C"):
        if not isinstance(env, SelfPlayWrapper):
            try: # if the env is a sb3-based env
                if not env.env_is_wrapped(wrapper_class=SelfPlayWrapper, indices=0)[0]:
                    env = SelfPlayWrapper(env)
            except AttributeError: # if the env is not a sb3-based env
                env = SelfPlayWrapper(env)

        if not os.path.exists(models_bank_path):
            os.makedirs(models_bank_path)
            print("Created the directory:", models_bank_path)
        else:
            for item in os.listdir(models_bank_path):
                item_path = os.path.join(models_bank_path, item)
                os.remove(item_path)

            print("Cleared all the files in the directory:", models_bank_path)

        print()

        template_model_name = "pao_model"

        clone_model_path = os.path.join(models_bank_path, template_model_name+str(0))
        clone_model.save(clone_model_path)

        models_bank = deque(maxlen=models_bank_maxlen)
        models_bank.append(clone_model_path)

        for epoch in range(1, self_play_epochs+1):
            print(f"\rWorking on epoch: #{epoch}", end='')

            opponent_model_idx = random.randint(0, len(models_bank)-1)
            opponent_model_path = models_bank[opponent_model_idx]
            opponent_model = self.load_model(opponent_model_path, 
                                            algorithm=algorithm, 
                                            force=True, dont_save=True)

            try:
                env.set_opponent_model(model=opponent_model)
            except AttributeError:
                env.env_method(method_name="set_opponent_model", model=opponent_model)

            self.model.learn(total_timesteps=iterations, 
                            log_interval=log_interval, 
                            reset_num_timesteps=False)

            model_epoch_path = os.path.join(models_bank_path, template_model_name+str(epoch))
            self.model.save(model_epoch_path)
            models_bank.append(model_epoch_path)

    @staticmethod
    def create_gif(frames, file_path):
        frame_images = [Image.fromarray(frame) for frame in frames]

        frame_images[0].save(
            file_path, format="GIF",
            append_images=frame_images,
            save_all=True,
            duration=90,
            loop=0
        )

    @staticmethod
    def load_train_configs(config_file_path): # TODO: make it more generic for the config files (for example: 2 or 3 dicts for any args)
        print("* Loading configs for training from", config_file_path)
        with open(config_file_path) as stream:
            configs = yaml.safe_load(stream)
        print('*')
        print("* Loaded configs from the file are:")
        print(configs)
        print()

        algorithm = list(configs.keys())[0].upper()
        configs = configs[algorithm]

        if algorithm == "DQN":
            iterations = configs["iterations"]
            iter_type = configs["iter_type"]
            n_step = configs["n_step"]
            target_soft_update = configs["target_soft_update"]
            target_update_interval = iterations * 0.15 * (0.5 if target_soft_update else 1)

            gamma = configs["gamma"] 
            buffer_size = configs["buffer_size"] 
            fill_buffer_episodes = configs["fill_buffer_episodes"]

            batch_size = configs["batch_size"]
            lr = configs["lr"]

            epsilon_active_portion = configs["epsilon_active_portion"]

            save_model_interval_portion = configs["save_model_interval_portion"]
            save_model_reward_threshold = configs["save_model_reward_threshold"]

            params = {
                "iterations": iterations,
                "iter_type": iter_type,
                "n_step": n_step,
                "soft_update": target_soft_update,
                "target_update_interval": target_update_interval,
                "gamma": gamma,
                "batch_size": batch_size,
                "warmup": fill_buffer_episodes,
                "epsilon_fn": configs["epsilon_fn"],
                "epsilon_active_portion": epsilon_active_portion,
                "save_model_interval_portion": save_model_interval_portion,
                "save_model_reward_threshold": save_model_reward_threshold,
            }

            kwargs = {
                "make_env": {**configs["make_env"]},
                "DQN": {"lr": lr, "buffer_size": buffer_size}
            }
        elif algorithm == "A2C":
            kwargs = {
                "n_env": configs["n_env"],
                "make_env": {**configs["make_env"]},
                "model_file_name": configs["model_file_name"],
                "verbose": configs["log_interval"],
                "log_interval": configs["log_interval"],
            }

            del configs["n_env"]
            del configs["make_env"]
            del configs["model_file_name"]
            del configs["verbose"]
            del configs["log_interval"]
            del configs["algorithm"]

            params = {**configs}
            
        return {"algorithm": algorithm, "params": params, "kwargs": kwargs}

    @staticmethod
    def load_run_configs(config_file_path): # TODO: make it more generic for the config files (for example: 2 or 3 dicts for any args)
        print("* Loading configs for running from", config_file_path)
        with open(config_file_path) as stream:
            configs = yaml.safe_load(stream)
        print('*')
        print("* Loaded configs from the file are:")
        print(configs)
        print()

        algorithm = list(configs.keys())[0].upper()
        configs = configs[algorithm]

        size = configs["make_env"]["size"].replace('(', '').replace(')', '').replace(' ', '').split(',')
        size = [int(i) for i in size]
        del configs["make_env"]["size"]

        kwargs = {
            "make_env": {**configs["make_env"], "size": size},
            "model_file_name": configs["model_file_name"],
            "gif_file_name": configs["gif_file_name"],
        }

        del configs["make_env"]
        del configs["model_file_name"]
        del configs["gif_file_name"]

        params = {**configs}

        return {"algorithm": algorithm, "params": params, "kwargs": kwargs}

    def load_model(self, model_path, algorithm, force=False, dont_save=False):
        if self.model is not None and not force:
            print("Load failed due to not forcing the load. Continuing to use the previous loaded model.")
            return None

        if algorithm == "A2C":
            try:
                from stable_baselines3 import A2C


                model = A2C.load(model_path)
            except ModuleNotFoundError:
                    print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                    return None
        elif algorithm == "PPO":
            try:
                from stable_baselines3 import PPO

                
                model = PPO.load(model_path)
            except ModuleNotFoundError:
                    print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                    return None            
        elif algorithm == "DQN":
            from rl.dqn import DQN


            model = DQN()
            model.load(model_path)

        if dont_save:
            return model
        else:
            self.set_model(model)

            return model

    def drop_model(self):
        self.model = None

    def set_model(self, model):
        self.model = model
 
    def get_model(self):
        return self.model

    def set_env(self, env):
        self.env = env

    def predict(self, obs, use_rules=False, deterministic=False, **kwargs):
        if self.model:
            action, _ = self.model.predict(obs, deterministic=deterministic, **kwargs)

            return action
        else:
            if use_rules:
                return self.run_rules_one_step(obs)
            else:
                print("No model has been loaded, and the use_rules arg was set to false.")

                return None

    def train(self, config_file_path, do_save=True):
        configs = Agent.load_train_configs(config_file_path)

        if configs["algorithm"] == "DQN":
            from rl.dqn import DQN
            from rl.train_utils import plot_train_history


            env = make_env(**configs["kwargs"]["make_env"])
            self.model = DQN(env=env, **configs["kwargs"]["DQN"])

            rewards, total_loss = self.model.run_training(**configs["params"])
            plot_train_history(rewards, total_loss)
        elif configs["algorithm"] == "A2C":
            try:
                from stable_baselines3.common.env_util import make_vec_env
                from stable_baselines3 import A2C
            except ModuleNotFoundError:
                print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                return None


            vec_env = make_vec_env(lambda: make_env(**configs["kwargs"]["make_env"]), n_envs=configs["kwargs"]["n_env"])
            self.model = A2C("MlpPolicy", vec_env,  **configs["params"], tensorboard_log=f"./logs/A2C")

            if configs["kwargs"]["make_env"].get("team_num", 1) >= 2:
                self.model.learn(total_timesteps=configs["kwargs"]["iterations"], log_interval=configs["kwargs"]["log_interval"])
            else:
                model_copy = A2C("MlpPolicy", vec_env,  **configs["params"])
                self.train_with_self_play(vec_env, model_copy, **configs["kwargs"])

        vec_env.close()
        if do_save:
            model_path = os.path.join("./models", configs["algorithm"], configs["kwargs"]["model_file_name"])
            self.model.save(model_path)

        return self.model

    def run(self, config_file_path):
        configs = Agent.load_run_configs(config_file_path)

        model_file_name = configs["kwargs"]["model_file_name"]
        if os.path.isfile(model_file_name):
            model_path = model_file_name
        else:
            model_path = os.path.join("./models", configs["algorithm"], model_file_name)

        self.load_model(model_path=model_path, algorithm=configs["algorithm"])

        env = make_env(**configs["kwargs"]["make_env"])
        if self_play_wrapper:=configs["kwargs"]["make_env"].get("self_play_wrapper", None) is not None:
            if self_play_wrapper:
                env.set_opponent_model(self.model)

        _, frames = self.run_agent_one_episode(env, **configs["params"])

        env.close()

        if configs["kwargs"].get("gif_file_name", None) is not None:
            gif_file_path = os.path.join("./gifs", configs["kwargs"]["gif_file_name"])
            Agent.create_gif(frames, gif_file_path)

            print()
            print(".gif file saved to:", gif_file_path)
        else:
            print()
            print("Skipped creating .gif file since no path was provided to save it.")