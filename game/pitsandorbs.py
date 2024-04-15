import pygame

import numpy as np

import os


class PitsAndOrbs:
    DIRECTIONS = ["0.west", "1.north", "2.east", "3.south"]
    CELLS = ["0.nothing", "1.player", "2.orb", "3.pit", "4.player&orb", 
            "5.player&pit", "6.orb&pit", "7.player&orb&pit",
            "8.out of bound"]
    ACTIONS = ["0.turn right", "1.move forward", "2.pick orb up", 
            "3.put orb down"]

    def __init__(self, size=(5, 5), orb_num=5, pit_num=5, seed=None, pygame_mode=True, pygame_with_help=True):
        assert len(size) == 2
        self.size = size

        assert size[0]*size[1] > (orb_num+pit_num)
        self.orb_num = orb_num
        self.pit_num = pit_num

        self.epsilon = 1e-5
        self.frame_time = 1 / 60

        self._pygame_mode = pygame_mode
        self._pygame_with_help = pygame_with_help
        if pygame_mode:
            pygame.init()

            self.multiplier = 100
            self.border_color = (255, 120, 0)
            self.border_width = 3
            self.border_margin = self.multiplier - (1.5 * self.border_width)

            self.play = self.play2

            self.screen_size = (
                self.size[0]*self.multiplier, 
                (self.size[1]+(3 if self._pygame_with_help else 0))*self.multiplier
            )
            self.screen = pygame.display.set_mode(size=self.screen_size, 
                                                flags=pygame.RESIZABLE)
            self.screen_rect = self.screen.get_rect()
            self.screen_dims = self.screen_rect.size

            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, int(0.5*self.multiplier))
            self.smaller_font = pygame.font.SysFont(None, int(0.15*self.multiplier))

            pygame.display.set_caption("Pits and Orbs")
        else:
            self.play = self.play1

        self.reset(seed=seed)

    def play1(self, show_obs_or_state_or_both=0, show_help=True, clear=True): # input=4 quits the game
        rewards = 0

        while True:
            self.show_board(show_obs_or_state_or_both=show_obs_or_state_or_both, 
                            show_help=show_help, clear=clear)

            action = int(input("Next Action: "))
            if action == 4:
                print("Quiting the game ...")
                break

            _, reward, done, _ = self.step(action)
            rewards += reward

            print("Taken reward for last action:", reward)
            print("All rewards until now:", rewards)
            print()

            if done:
                print(f"Game ended successfully with {self.player_movements} movements.")
                break

    def play2(self, show_obs_or_state_or_both=0, show_help=False, clear=False, print_is_enabled=True): # play function for pygame
        obs = None
        rewards = 0
        done = False
        info = None
        if print_is_enabled:
            self.show_board(obs=obs, info=info, 
                            show_obs_or_state_or_both=show_obs_or_state_or_both, 
                            show_help=show_help, clear=clear)

        while True:
            self._update_screen()

            action = self._check_events()

            if action is not None and not done:
                obs, reward, done, info = self.step(action)
                rewards += reward

                if print_is_enabled:
                    self.show_board(obs=obs, info=info, 
                                    show_obs_or_state_or_both=show_obs_or_state_or_both, 
                                    show_help=show_help, clear=clear)

                    print("Taken reward for last action:", reward)
                    print("All rewards until now:", rewards)
                    print()

                    if done and not self.printed_game_is_finished:
                        print(f"Game ended successfully with {self.player_movements} movements.")
                        self._build_finish_materials()
                        self.printed_game_is_finished = True

            self.clock.tick(1/self.frame_time)

    def _check_events(self):
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                if width != self.screen_size[0]:
                    width = self.screen_size[0]
                if height < self.size[1]*self.multiplier:
                    height = self.size[1]*self.multiplier
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_0:
                        action = 0
                    case pygame.K_1:
                        action = 1
                    case pygame.K_2:
                        action = 2
                    case pygame.K_3:
                        action = 3
                    case pygame.K_q:
                        pygame.quit()
                        quit()                      

        return action

    def _update_screen(self):
        self.screen.fill((200, 200, 200))

        self._display_objects()

        self._draw_table()

        if self._pygame_with_help:
            self._draw_info_and_help()

        if self.printed_game_is_finished:         
            self.screen.blit(self.finished_text, self.finished_text_rect)

        pygame.display.flip()

    def _display_objects(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                cell_type = self.board_state[i, j]
                if cell_type == 0:
                    continue
                elif cell_type == 1:
                    self._draw_player(i, j)
                    self._draw_player_direction(i, j)
                elif cell_type == 2:
                    self._draw_orb(i, j)
                elif cell_type == 3:
                    self._draw_pit(i, j)
                elif cell_type == 4:
                    self._draw_player(i, j)
                    self._draw_orb(i,j)
                    self._draw_player_direction(i, j)
                elif cell_type == 5:
                    self._draw_player(i, j)
                    self._draw_pit(i,j)
                    self._draw_player_direction(i, j)
                elif cell_type == 6:
                    self._draw_pit(i, j)
                    self._draw_orb(i, j, half_size=True, center=True)
                elif cell_type == 7:
                    self._draw_player(i, j)
                    self._draw_pit(i, j)
                    self._draw_orb(i, j, half_size=True, center=True)
                    self._draw_player_direction(i, j)

    def _draw_player(self, i, j):
        rect = pygame.draw.rect(
            self.screen, 
            (0, 0, 255), 
            (j*self.multiplier+self.border_width, 
            i*self.multiplier+self.border_width, 
            self.border_margin, self.border_margin)
        )

        if self.player_has_orb:
            self._draw_orb(i, j, half_size=True)

        return rect

    def _draw_player_direction(self, i, j):
        match self.player_direction:
            case 0:
                points = [
                (j*self.multiplier+self.border_width+self.multiplier/4, i*self.multiplier+self.border_width+self.multiplier/4),
                (j*self.multiplier+self.border_width+self.multiplier/4, i*self.multiplier+self.border_width+self.multiplier*(3/4)),
                (j*self.multiplier+self.border_width+self.multiplier/10, i*self.multiplier+self.border_width+self.multiplier/2),
                ]
            case 1:
                points = [
                (j*self.multiplier+self.border_width+self.multiplier/4, i*self.multiplier+self.border_width+self.multiplier/4),
                (j*self.multiplier+self.border_width+self.multiplier*(3/4), i*self.multiplier+self.border_width+self.multiplier/4),
                (j*self.multiplier+self.border_width+self.multiplier/2, i*self.multiplier+self.border_width+self.multiplier/10),
                ]
            case 2:
                points = [
                (j*self.multiplier+self.border_width+self.multiplier*(3/4), i*self.multiplier+self.border_width+self.multiplier/4),
                (j*self.multiplier+self.border_width+self.multiplier*(3/4), i*self.multiplier+self.border_width+self.multiplier*(3/4)),
                (j*self.multiplier+self.border_width+self.multiplier*(9/10), i*self.multiplier+self.border_width+self.multiplier/2),
                ]
            case 3:
                points = [
                (j*self.multiplier+self.border_width+self.multiplier/4, i*self.multiplier+self.border_width+self.multiplier*(3/4)),
                (j*self.multiplier+self.border_width+self.multiplier*(3/4), i*self.multiplier+self.border_width+self.multiplier*(3/4)),
                (j*self.multiplier+self.border_width+self.multiplier/2, i*self.multiplier+self.border_width+self.multiplier*(9/10)),
                ]

        rect = pygame.draw.polygon(
            self.screen, 
            (255, 0, 255), 
            points, 
        )

        return rect

    def _draw_orb(self, i, j, half_size=False, center=False):
        if half_size:
            divisor = 2
            color = (255, 255, 0)
        else:
            divisor = 1
            color = (0, 255, 0)

        if not center:
            push_x = 0
            push_y = 0
        else:
            push_x = self.border_margin // (2 * divisor)
            push_y = self.border_margin // (2 * divisor)

        rect = pygame.draw.ellipse(
            self.screen, 
            color, 
            (j*self.multiplier+self.border_width+2+push_x, 
            i*self.multiplier+self.border_width+2+push_y, 
            (self.border_margin-4)//divisor, (self.border_margin-4)//divisor)
        )

        return rect

    def _draw_pit(self, i, j):
        rect = pygame.draw.ellipse(
            self.screen, 
            (255, 0, 0), 
            (j*self.multiplier+self.border_width+2, 
            i*self.multiplier+self.border_width+10, 
            self.border_margin-4, self.border_margin-20)
        )

        return rect

    def _draw_table(self):
        for i in range(self.size[0]+1):
            pygame.draw.line(self.screen, self.border_color, 
                            (0, i*self.multiplier), 
                            (self.screen_dims[0], i*self.multiplier), 
                            width=self.border_width)

        for j in range(self.size[1]+1):
            margin = -1 if j == self.size[1] else 0
            pygame.draw.line(self.screen, self.border_color, 
                            (j*self.multiplier+margin, 0), 
                            (j*self.multiplier+margin, self.screen_dims[0]), 
                            width=self.border_width)

    def _draw_info_and_help(self):
        player_movement_txt = self.font.render(f"Player Movements: #{self.player_movements}", True, (0, 0, 0), (200, 200, 200))
        player_movement_rect = player_movement_txt.get_rect()
        player_movement_rect.topleft = (0, self.size[0]*self.multiplier+self.border_width+10)
        self.screen.blit(player_movement_txt, player_movement_rect)

        help_txt = self.smaller_font.render(
            f"{'    |    '.join([str(i)+' ==> '+action.split('.')[-1].upper() for i, action in enumerate(PitsAndOrbs.ACTIONS)])}", 
            True, 
            (0, 0, 0), 
            (200, 200, 200)
        )
        help_rect = help_txt.get_rect()
        help_rect.topleft = (player_movement_rect.bottomleft[0]+self.border_width, 
                            player_movement_rect.bottomleft[1]+self.multiplier/4)
        self.screen.blit(help_txt, help_rect)

        for i, cell_type in enumerate(PitsAndOrbs.CELLS):
            type_ = cell_type.split('.')[-1]
            if i == 0:
                continue
            elif i == 1:
                obj_rect = self._draw_player(self.size[0]+1, 0)
            elif i == 2:
                obj_rect = self._draw_orb(self.size[0]+1, 1)
            elif i == 3:
                obj_rect = self._draw_pit(self.size[0]+1, 2)
            elif i == 4:
                obj_rect = self._draw_player(self.size[0]+1, 3)
                self._draw_orb(self.size[0]+1, 3)
            elif i == 5:
                obj_rect = self._draw_player(self.size[0]+1, 4)
                self._draw_pit(self.size[0]+1, 4)
            elif i == 6:
                obj_rect = self._draw_pit(self.size[0]+2, 0)
                self._draw_orb(self.size[0]+2, 0, half_size=True, center=True)
            elif i == 7:
                obj_rect = self._draw_player(self.size[0]+2, 1)
                self._draw_pit(self.size[0]+2, 1)
                self._draw_orb(self.size[0]+2, 1, half_size=True, center=True)

            self._write_text(type_, obj_rect)

    def _write_text(self, text, obj_rect):
        text = self.smaller_font.render(text, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = obj_rect.center

        self.screen.blit(text, text_rect)

    def _build_finish_materials(self):
        text = pygame.font.SysFont(None, int(0.85*self.multiplier)).render("Round Finished!", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = self.screen.get_rect().center

        text.set_alpha(175)

        self.finished_text = text
        self.finished_text_rect = text_rect

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

    def step(self, action):
        reward = self._do_action(action)

        obs = self.get_observation()
        reward += self.compute_current_reward()
        done = self.is_done()
        info = self.get_info()

        return obs, reward, done, info

    def reset(self, seed=None):
        np.random.seed(seed)

        self.printed_game_is_finished = False

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
        for player_cell in (1, 4, 5, 7): # all possible values for player being in a cell
            i, j = np.where(self.board_state==player_cell)
            if i.shape != (0,) and j.shape != (0,):
                break

        return i[0], j[0]

    def clear_screen(self):
        os.system("cls" if os.name=="nt" else "clear")

    def show_board(self, obs=None, info=None, show_obs_or_state_or_both=0, show_help=True, clear=True):
        obs = obs if obs is not None else self.get_observation()
        info = info if info is not None else self.get_info()

        if clear:
            self.clear_screen()

        if show_obs_or_state_or_both == 0:
            print(obs)
            print()
        elif show_obs_or_state_or_both == 1:
            print(self.board_state)
            print()
        elif show_obs_or_state_or_both == 2:
            print(self.board_state)
            print()
            print(obs)
            print()
        
        print(info)
        print()

        if show_help:
            print("Directions:", PitsAndOrbs.DIRECTIONS)
            print("Cell Types:", PitsAndOrbs.CELLS)
            print("Actions:", PitsAndOrbs.ACTIONS)
            print()
       
    def close(self):
        if self._pygame_mode:
            self.board_state = None
    
            pygame.quit()


if __name__ == "__main__":
    game = PitsAndOrbs()
    game.player_direction = 1
    game.play()