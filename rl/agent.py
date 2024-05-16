import yaml

import time

from PIL import Image

import os

from utils import make_env


class Agent:

    def __init__(self, model_path=None, model=None):
        assert model_path is None and model is not None
        assert model_path is not None and model is None

        self.drop_model()

        if model_path:
            self.load_model(model_path)

        if model:
            self.model = model

    def _run_rules_one_step(self, obs):
        return None

    def _run_agent_one_episode(self, env, max_steps=500, fps=None, 
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
    def load_train_configs(config_file_path):
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
    def load_run_configs(config_file_path):
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

    def load_model(self, model_path, algorithm, force=False):
        if self.model is not None:
            print("Warning! Object already has a loaded model.")
            if not force:
                print("Load failed due to not forcing the load. Continuing to use the previous loaded model.")

                return

        if model_path.endswith(".zip"):
            try:
                from stable_baselines3 import A2C, PPO
            except:
                print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                return None


            if algorithm == "A2C":
                self.model = A2C.load(model_path)
            elif algorithm == "PPO":
                self.model = PPO.load(model_path)

        elif model_path.endswith(".h5") \
            or model_path.endswith(".keras") \
            or model_path.endswith("/"):
            from rl.dqn import DQN


            if algorithm == "DQN":
                self.model = DQN()
                self.load_model(model_path)
            else:
                raise Exception("Wrong algorithm for a tensorflow/keras model.")

    def drop_model(self):
        self.model = None

    def predict(self, obs, use_rules=False, deterministic=False, **kwargs):
        if self.model:
            action, _ = self.model.predict(obs, deterministic=deterministic, **kwargs)

            return action
        else:
            if use_rules:
                return self._run_rules_one_step(obs)
            else:
                print("No model has been loaded, and the use_rules arg was set to false.")

                return None

    def train(self, config_file_path, do_save=True):
        configs = Agent.load_train_configs(config_file_path)

        if configs["algorithm"] == "DQN":
            from agent.dqn import DQN
            from agent.train_utils import plot_train_history


            env = make_env(**configs["kwargs"]["make_env"])
            self.model = DQN(env=env, **configs["kwargs"]["DQN"])

            rewards, total_loss = self.model.run_training(**configs["params"])
            plot_train_history(rewards, total_loss)
        elif configs["algorithm"] == "A2C":
            try:
                from stable_baselines3.common.env_util import make_vec_env
                from stable_baselines3 import A2C
            except:
                print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                return None


            vec_env = make_vec_env(lambda: make_env(**configs["kwargs"]["make_env"]), n_envs=configs["kwargs"]["n_env"])
            self.model = A2C("MlpPolicy", vec_env,  **configs["params"], tensorboard_log=f"./logs/A2C")

            self.model.learn(total_timesteps=configs["kwargs"]["iterations"], log_interval=configs["kwargs"]["log_interval"])

        vec_env.close()
        if do_save:
            model_path = os.path.join("./models", configs["algorithm"], configs["kwargs"]["model_file_name"])
            self.model.save(model_path)

        return self.model

    def run(self, config_file_path):
        configs = Agent.load_run_configs(config_file_path)

        model_path = os.path.join("./models", configs["algorithm"], configs["kwargs"]["model_file_name"])
        self.load_model(model_path=model_path, algorithm=configs["algorithm"])

        env = make_env(**configs["kwargs"]["make_env"])

        _, frames = self._run_agent_one_episode(env, **configs["params"])

        env.close()

        if configs["kwargs"].get("gif_file_name", None) is not None:
            gif_file_path = os.path.join("./gifs", configs["kwargs"]["gif_file_name"])
            Agent.create_gif(frames, gif_file_path)

            print()
            print(".gif file saved to:", gif_file_path)
        else:
            print()
            print("Skipped creating .gif file since no path was provided to save it.")