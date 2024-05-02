import gym


class StepsLimit(gym.Wrapper):
        
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps

        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        
        truncated = True if self.current_step>=self.max_steps else False

        state, reward, done, info = self.env.step(action)
        done = done or truncated
        
        return state, reward, done, info