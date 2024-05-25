import numpy as np

try:
    from game.player import Player
    from game.memory import Memory
except:
    from player import Player
    from memory import Memory


class Team:

    def __init__(self, game, players_index):
        self.game = game

        # initiate multiple players
        self._player_turn = 0 # (0, ..., self.player_num-1)
        self._players = [Player(game, player_index) for player_index in players_index]
        self._players_num = len(self._players)

        self._filled_pits = []

        # update shared memory according to every players' neighbors
        if self.game._return_obs_type == "partial obs":
            self._memory = Memory(game)
            for i in range(self._players_num): 
                self._player_turn = i
                neighbors = self.get_neighbors()
                self.update_memory(neighbors)
            self._player_turn = 0

    @property
    def current_player(self):
        return self._players[self._player_turn]

    @property
    def players_pos(self):
        poses = [player.position for player in self._players]

        return poses

    @property
    def players(self):
        return self._players

    @property
    def player_turn(self):
        return self._player_turn

    @property
    def scores(self):
        return len(self._filled_pits)

    @property
    def filled_pits(self):
        return self._filled_pits

    def add_to_filled_pits(self, pit_pos):
        self._filled_pits.append(pit_pos)

    def rem_from_filled_pits(self, pit_pos):
        self._filled_pits.remove(pit_pos)

    def get_filled_pits_positions(self):
        not_found_index = max(self.game.size)

        pits_positions = []
        for pit_pos in self.game._pits_pos:
            if pit_pos in self._filled_pits:
                pits_positions.append(pit_pos)
            else:
                pits_positions.append((not_found_index, not_found_index))

        return pits_positions

    def update_memory(self, neighbors):
        self._memory.update(neighbors, self.current_player.position)

    def get_memory(self):
        return self._memory.get()

    def get_neighbors(self):
        padded_board_state = np.zeros((self.game.size[0]+2, self.game.size[1]+2), dtype=np.uint8)
        padded_board_state[0, :] = len(self.game.CELLS)
        padded_board_state[-1, :] = len(self.game.CELLS)
        padded_board_state[:, 0] = len(self.game.CELLS)
        padded_board_state[:, -1] = len(self.game.CELLS)
        padded_board_state[1:-1, 1:-1] = self.game.board_state

        player_pos_i, player_pos_j = self.current_player.position
        player_pos_i += 1
        player_pos_j += 1
        obs = padded_board_state[player_pos_i-1: player_pos_i+2, player_pos_j-1: player_pos_j+2]

        return obs

    def change_player_turn(self):
        prev_player_turn = self._player_turn
        self._player_turn = (prev_player_turn + 1) % self._players_num

        change_team = (prev_player_turn != 0 and self._player_turn == 0) or (self._players_num == 1)

        return change_team