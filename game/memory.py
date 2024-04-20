import numpy as np


class Memory:

    def __init__(self, game):
        self.board = np.zeros((game.size[0]+2, game.size[1]+2), 
                                    dtype=np.uint8)

    def update(self, neighbors, player_pos):
        player_pos_i, player_pos_j = player_pos
        player_pos_i += 1
        player_pos_j += 1

        self.board[player_pos_i-1: player_pos_i+2, 
                            player_pos_j-1: player_pos_j+2] = neighbors

    def get(self):
        board_withoud_bounds = self.board[1: -1, 1: -1].copy()

        return board_withoud_bounds