# import numpy as np


# class Agent:

#     def __init__(self, env):
#         self.env = env

#         self.env_size = env.size
#         self.board_memory = np.array((env.size[0]+2, env.size[1]+2), 
#                                     dtype=np.uint8)

#     def predict(self, obs, info):
#         self.board_memory = self.env.board_state.copy() # self._update_memory(obs, info)

#         if player in orb ...
        
#         optimal_orb_pos = self._optimal_orb_choice_from_agent_to_pits_dist(info)
#         direction_to_optimal_orb_pos = self.calc_direction_to(optimal_orb_pos, info)
#         self._move_player_to_direction(direction_to_optimal_orb_pos, info)

#         return action

#     def _update_memory(self, obs, info):
#         player_pos_i, player_pos_j = info["player position"]
#         player_pos_i += 1
#         player_pos_j += 1

#         self.board_memory[player_pos_i-1:player_pos_i+2, 
#                             player_pos_j-1:player_pos_j+2] = obs
        
#     def _optimal_orb_choice_from_agent_to_pits_dist(self, info):
#         player_pos = info["player position"]

#         min_dists = []
#         orb_pos_list = []
#         for orb_cell in (2, 4): # for all orbs
#             orb_pos_Is, orb_pos_Js = np.where(self.board_memory==orb_cell)
#             if len(orb_pos_Is) < 1:
#                 continue

#             for orb_pos in zip(orb_pos_Is, orb_pos_Js):
#                 orb_pos_list.append(orb_pos)
#                 agent_to_orb_dist = Agent.calc_manhatan_dist(orb_pos, player_pos)

#                 orb_to_pit_dists = []
#                 for pit_cell in (3, 5): # for all pits
#                     pit_pos_Is, pit_pos_Js = np.where(self.board_memory==pit_cell)
#                     if len(pit_pos_Is) < 1:
#                         continue

#                     for pit_pos in zip(pit_pos_Is, pit_pos_Js):
#                         orb_to_pit_dists.append(Agent.calc_manhatan_dist(orb_pos, pit_pos))

#                 min_dists.append(agent_to_orb_dist+min(orb_to_pit_dists))

#         orb_pos_and_min_dists = list(zip(orb_pos_list, min_dists))
#         orb_pos_and_min_dists = sorted(orb_pos_and_min_dists, key=lambda x: x[1])
#         optimal_orb_choice = orb_pos_and_min_dists[0][0]

#         return optimal_orb_choice

#     def _move_player_to_direction(self, direction, info):
#         direction_types = self.env.DIRECTIONS

#         player_direction = info["player direction"]
#         player_direction_index = direction_types.index(player_direction)
#         direction_index = direction_types.index(direction)

        

#     @staticmethod
#     def calc_direction_to(destination_pos, info):
#         player_pos_i, player_pos_j = info["player position"]

#         i_diff = player_pos_i - destination_pos[0]
#         j_diff = player_pos_j - destination_pos[1]

#         if i_diff > 0 and j_diff > 0:
#             pass
#         elif i_diff > 0 and j_diff > 0:
#             pass
#         elif i_diff > 0 and j_diff > 0:
#             pass
#         elif i_diff > 0 and j_diff > 0:
#             pass

#     @staticmethod
#     def calc_manhatan_dist(pos1, pos2):
#         dist = abs(pos1[0] - pos2[0])
#         dist += abs(pos1[1] - pos2[1])

#         return dist


class Agent:

    def __init__(self, env, model_path=None):
        self.env = env

        if model_path:
            self.load_model(model_path)

    def predict(self, obs, deterministic=False):
        if self.model:
            return self.run_rl_model(obs, deterministic)
        else:
            return self.run_classical(obs)

    def run_rl_model(self, obs, deterministic=False):
        action, _ = self.model.predict(obs, deterministic=deterministic)

        return action

    def run_classical(self, obs):
        return None

    def load_model(self, model_path):
        try:
            from stable_baselines3 import A2C
        except:
            print("Stable Baselines 3 is needed to run a deep RL model.")
            return None

        self.model = A2C.load(model_path)

