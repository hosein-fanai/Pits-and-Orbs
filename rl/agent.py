import yaml

import time

from PIL import Image

import os

from collections import deque

import random

from utils import make_env

from wrappers.self_play_wrapper import SelfPlayWrapper


class Agent:

    def __init__(self, model_path=None, model=None):
        assert model_path is None or model is None

        self.drop_model()

        if model_path:
            self.load_model(model_path)

        if model:
            self.model = model

    def run_rules_one_step(self, obs):
        return None

    def run_agent_one_episode(self, env, max_steps=500, fps=None, 
                deterministic_action_choice=False, 
                print_rewards=True, return_frames=True):
        assert return_frames and env.game._pygame_mode

        obs = env.reset()
        done = False
        rewards = 0.

        frames = []
        frames_counter = 0

        start = time.time()
        for step in range(max_steps):
            action = self.predict(obs, deterministic=deterministic_action_choice)

            obs, reward, done, info = env.step(action)

            rewards += reward
            frames_counter += 1

            if print_rewards:
                print(f"\rTotal Rewards: {rewards:.4f}, Current Reward: {reward:.4f}, Taken Action: {action}", end="")

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
        print("Total rewards were:", rewards)
        print("Final info:", info)
        print("The episode was fps:", int(frames_counter/(time.time()-start)))
        print("The episode length was:", step+1)

        return rewards, frames

    def train_with_self_play(self, env, clone_model, models_bank_maxlen=5, self_play_epochs=20, 
                            iterations=5_000_000, log_interval=1_000, models_bank_path="./models/self-play", 
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