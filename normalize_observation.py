import gym


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation / (len(self.env.game.CELLS)-1)

