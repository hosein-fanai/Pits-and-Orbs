import numpy as np

import os


class PitsAndOrbs:
    DIRECTIONS = ["0.west", "1.north", "2.east", "3.south"]
    CELLS = ["0.nothing", "1.player", "2.orb", "3.pit", "4.player&orb", 
            "5.player&pit", "6.orb&pit", "7.player&orb&pit",
            "8.out of bound"]
    ACTIONS = ["0.turn right", "1.move forward", "2.pick orb up", 
            "3.put orb down"]

    def __init__(self, size=(5, 5), orb_num=5, pit_num=5, seed=None):
        assert len(size) == 2
        self.size = size

        assert size[0]*size[1] > (orb_num+pit_num)
        self.orb_num = orb_num
        self.pit_num = pit_num

        self.reset(seed=seed)

    def step(self, action):
        reward = self._do_action(action)

        obs = self.get_observation()
        reward += self.compute_current_reward()
        done = self.is_done()
        info = self.get_info()

        return obs, reward, done, info

    def reset(self, seed=None):
        np.random.seed(seed)

        self.board_state = np.zeros(self.size, dtype=np.uint8)
        indices = list(range(self.board_state.size))

        random_cells = np.random.choice(indices, size=self.pit_num+self.orb_num+1, replace=False)
        pit_indices = random_cells[: self.pit_num]
        orb_indices = random_cells[self.pit_num: self.pit_num+self.orb_num]
        player_index = random_cells[-1]

        self.board_state = self.board_state.flatten()

        # put the player randomly in the board
        self.board_state[player_index] = 1

        # create random pits and orbs in the board
        self.board_state[orb_indices] = 2
        self.board_state[pit_indices] = 3
        
        self.board_state = self.board_state.reshape(self.size)

        self.player_direction = np.random.randint(len(PitsAndOrbs.DIRECTIONS))
        self.player_has_orb = False
        self.player_movements = 0

        obs = self.get_observation()
        info = self.get_info()

        return obs, info

    def _do_action(self, action):
        reward = 0

        match action:
            case 0: # turn right
                reward = self._turn_right(reward)                
            case 1: # move forward
                reward = self._move_forward(reward)
            case 2: # pick orb up
                reward = self._pick_orb_up(reward)
            case 3: # put orb down
                reward = self._put_orb_down(reward)
            case _: # not valid action
                print("Not a valid action!")
                raise

        return reward

    def _turn_right(self, reward):
        self.player_direction = (self.player_direction + 1) % len(PitsAndOrbs.DIRECTIONS)
        # reward -= 0.05

        return reward

    def _move_forward(self, reward):
        player_pos_prev = self.get_player_position()
        player_pos_i, player_pos_j = player_pos_prev

        match self.board_state[player_pos_i, player_pos_j]:
            case 1:
                self.board_state[player_pos_i, player_pos_j] = 0
            case 4:
                self.board_state[player_pos_i, player_pos_j] = 2
            case 5:
                self.board_state[player_pos_i, player_pos_j] = 3
            case 7:
                self.board_state[player_pos_i, player_pos_j] = 6
            case 9:
                self.board_state[player_pos_i, player_pos_j] = 8

        match self.player_direction:
            case 0: # west
                player_pos_j -= 1
            case 1: # north
                player_pos_i -= 1
            case 2: # east
                player_pos_j += 1
            case 3: # south
                player_pos_i += 1
        player_pos_i = max(0, min(player_pos_i, self.size[0]-1))
        player_pos_j = max(0, min(player_pos_j, self.size[1]-1))

        match self.board_state[player_pos_i, player_pos_j]:
            case 0:
                self.board_state[player_pos_i, player_pos_j] = 1
            case 2:
                self.board_state[player_pos_i, player_pos_j] = 4

                # reward += 0.1
            case 3:
                self.board_state[player_pos_i, player_pos_j] = 5
            case 6:
                self.board_state[player_pos_i, player_pos_j] = 7
            case 8:
                self.board_state[player_pos_i, player_pos_j] = 9

        if (player_pos_i, player_pos_j) != player_pos_prev: # player actually moved
            self.player_movements += 1

        # reward -= 0.05

        return reward

    def _pick_orb_up(self, reward):
        if self.player_has_orb:
            # reward -= 0.1

            return reward

        player_pos_i, player_pos_j = self.get_player_position()
        if self.board_state[player_pos_i, player_pos_j] == 4:
            self.board_state[player_pos_i, player_pos_j] = 1
            self.player_has_orb = True

            # reward += 0.1
        else:
            # reward -= 0.1
            pass

        return reward

    def _put_orb_down(self, reward):
        if not self.player_has_orb:
            # reward -= 0.1

            return reward

        player_pos_i, player_pos_j = self.get_player_position()

        match self.board_state[player_pos_i, player_pos_j]:
            case 1:
                self.board_state[player_pos_i, player_pos_j] = 4
                self.player_has_orb = False

                # reward -= 0.1
                pass
            case 4:
                # reward -= 0.05
                pass
            case 5:
                self.board_state[player_pos_i, player_pos_j] = 7
                self.player_has_orb = False

                self._move_orbs_randomly()

                reward += 1.
            case 7:
                # reward -= 0.1
                pass

        return reward

    def _move_orbs_randomly(self):
        for orb_cell in (2, 4):
            orb_pos_Is, orb_pos_Js = np.where(self.board_state==orb_cell)
            if len(orb_pos_Is) < 1:
                continue

            for orb_pos in zip(orb_pos_Is, orb_pos_Js):
                if np.random.rand() > 0.1: # don't move this orb
                    continue
                else: # randomly move this orb
                        direction = np.random.randint(len(PitsAndOrbs.DIRECTIONS))
                        self._move_orb(orb_pos, direction)

    def _move_orb(self, orb_pos, direction):
        orb_pos_i, orb_pos_j = orb_pos

        match self.board_state[orb_pos_i, orb_pos_j]:
            case 2:
                self.board_state[orb_pos_i, orb_pos_j] = 0
                prev_cell = 2
            case 4:
                self.board_state[orb_pos_i, orb_pos_j] = 1
                prev_cell = 4

        match direction:
            case 0: # west
                orb_pos_j -= 1
            case 1: # north
                orb_pos_i -= 1
            case 2: # east
                orb_pos_j += 1
            case 3: # south
                orb_pos_i += 1
        orb_pos_i = max(0, min(orb_pos_i, self.size[0]-1))
        orb_pos_j = max(0, min(orb_pos_j, self.size[1]-1))

        match self.board_state[orb_pos_i, orb_pos_j]:
            case 0:
                self.board_state[orb_pos_i, orb_pos_j] = 2
            case 1:
                self.board_state[orb_pos_i, orb_pos_j] = 4
            case 2:
                self.board_state[orb_pos] = prev_cell
            case 3:
                self.board_state[orb_pos_i, orb_pos_j] = 6
            case 4:
                self.board_state[orb_pos] = prev_cell
            case 5:
                self.board_state[orb_pos_i, orb_pos_j] = 7
            case 6:
                self.board_state[orb_pos] = prev_cell

    def get_observation(self):
        padded_board_state = np.zeros((self.size[0]+2, self.size[1]+2), dtype=np.uint8)
        padded_board_state[0, :] = len(PitsAndOrbs.CELLS) - 1
        padded_board_state[-1, :] = len(PitsAndOrbs.CELLS) - 1
        padded_board_state[:, 0] = len(PitsAndOrbs.CELLS) - 1
        padded_board_state[:, -1] = len(PitsAndOrbs.CELLS) - 1
        padded_board_state[1:-1, 1:-1] = self.board_state

        player_pos_i, player_pos_j = self.get_player_position()
        player_pos_i += 1
        player_pos_j += 1
        obs = padded_board_state[player_pos_i-1:player_pos_i+2, player_pos_j-1:player_pos_j+2]

        return obs

    def compute_current_reward(self):
        return 0

    def is_done(self):
        cell_type6_num = len(np.where(self.board_state==6)[0])
        cell_type7_num = len(np.where(self.board_state==7)[0])
        done = (cell_type6_num+cell_type7_num == self.orb_num)

        return done

    def get_info(self):
        return {
            "player position": self.get_player_position(),
            "player direction": PitsAndOrbs.DIRECTIONS[self.player_direction], 
            "player has orb": self.player_has_orb,
            "player movements#": self.player_movements,
            }
  
    def get_player_position(self):
        for player_cell in (1, 4, 5, 7, 9): # all possible values for player being in a cell
            i, j = np.where(self.board_state==player_cell)
            if i.shape != (0,) and j.shape != (0,):
                break

        return i[0], j[0]

    def clear_screen(self):
        os.system("cls" if os.name=="nt" else "clear")

    def show_board(self, show_obs=False, show_help=True, clear=True):
        if clear:
            self.clear_screen()

        if show_obs:
            print(self.get_observation())
        else:
            print(self.board_state)

        print(self.get_info())
        print()

        if show_help:
            print("Directions:", PitsAndOrbs.DIRECTIONS)
            print("Cell Types:", PitsAndOrbs.CELLS)
            print("Actions:", PitsAndOrbs.ACTIONS)
            print()

    def play(self, show_obs=False, show_help=True, clear=True):
        while True:
            self.show_board(show_obs=show_obs, show_help=show_help, clear=clear)

            action = int(input("Next Action: " ))
            if action == 4:
                break

            _, _, done, _ = self.step(action)
            
            if done:
                print(f"Game ended successfully with {self.player_movements} movements.")
                break


if __name__ == "__main__":
    game = PitsAndOrbs()
    game.play()