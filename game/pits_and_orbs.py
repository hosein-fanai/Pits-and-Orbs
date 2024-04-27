import pygame

import numpy as np

import sys

import os

try:
    from game.memory import Memory
except:
    from memory import Memory


class PitsAndOrbs:
    DIRECTIONS = ["0.west", "1.north", "2.east", "3.south"]
    CELLS = ["0.nothing", "1.player", "2.orb", "3.pit", "4.player&orb", 
            "5.player&pit", "6.orb&pit", "7.player&orb&pit"] # , "8.out of bound"
    ACTIONS = ["0.turn right", "1.move forward", "2.pick orb up", 
            "3.put orb down"]

    OBSERVATIONS = ["partial obs", "full state", "neighbors"] 

    FPS = 60

    def __init__(self, size=(5, 5), orb_num=5, pit_num=5, players_num=1, seed=None, 
                pygame_mode=True, pygame_with_help=True, return_obs_type="partial obs", 
                reward_function_type='0', reward_function=None):
        assert len(size) == 2
        self.size = size

        assert size[0]*size[1] > (orb_num+pit_num+players_num)
        self.orb_num = orb_num
        self.pit_num = pit_num
        self.players_num = players_num

        self.seed = seed
        self._pygame_mode = pygame_mode
        self._pygame_with_help = pygame_with_help

        assert return_obs_type.lower() in PitsAndOrbs.OBSERVATIONS
        self._return_obs_type = return_obs_type.lower()

        if reward_function is None:
            if reward_function_type == '0':
                self._reward_function = PitsAndOrbs._reward_function0
            elif reward_function_type == '1':
                self._reward_function = PitsAndOrbs._reward_function1
        else:
            self._reward_function = reward_function

    def _check_events(self):
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
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
                        sys.exit()                     

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
        if self._return_obs_type in ("partial obs", "neighbors"):
            board = self.memory.get()
        elif self._return_obs_type == "full state":
            board = self.board_state

        for player_index, (player_pos_i, player_pos_j) in enumerate(self.players_pos):
            self._draw_player(player_pos_i, player_pos_j, player_index)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                cell_type = board[i, j]
                if cell_type == 0:
                    continue
                elif cell_type == 1:
                    pass
                elif cell_type == 2:
                    self._draw_orb(i, j)
                elif cell_type == 3:
                    self._draw_pit(i, j)
                elif cell_type == 4:
                    self._draw_orb(i,j)
                elif cell_type == 5:
                    self._draw_pit(i,j)
                elif cell_type == 6:
                    self._draw_pit(i, j)
                    self._draw_orb(i, j, half_size=True, center=True)
                elif cell_type == 7:
                    self._draw_pit(i, j)
                    self._draw_orb(i, j, half_size=True, center=True)

        for player_index, (player_pos_i, player_pos_j) in enumerate(self.players_pos):
            self._draw_player_direction(player_pos_i, player_pos_j, player_index)

    def _draw_player(self, i, j, player_index, is_help=False):
        rect = pygame.draw.rect(
            self.screen, 
            (0, 0, 255), 
            (j*self.multiplier+self.border_width, 
            i*self.multiplier+self.border_width, 
            self.border_margin, self.border_margin)
        )

        if not is_help:
            player_index_text = self.smaller_font.render(str(player_index), True, (0, 0, 0), (200, 200, 200))
            player_index_rect = player_index_text.get_rect()
            player_index_rect.topright = rect.topright
            self.screen.blit(player_index_text, player_index_rect)

        if self.players_have_orb[player_index] and not is_help:
            self._draw_orb(i, j, half_size=True)

        return rect

    def _draw_player_direction(self, i, j, player_index):
        match self.players_direction[player_index]:
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
        for i in range(self.players_num):
            player_turn_text = ""
            if self.player_turn == i:
                player_turn_text = "    has to play now!"

            player_movement_txt = self.smaller_font.render(f"Player{i} Movements: #{self.players_movements[i]}"+player_turn_text, True, (0, 0, 0), (200, 200, 200))
            player_movement_rect = player_movement_txt.get_rect()
            player_movement_rect.topleft = (0, self.size[0]*self.multiplier+self.border_width+10+30*i)
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
                obj_rect = self._draw_player(self.size[0]+1, 0, 0, is_help=True)
            elif i == 2:
                obj_rect = self._draw_orb(self.size[0]+1, 1)
            elif i == 3:
                obj_rect = self._draw_pit(self.size[0]+1, 2)
            elif i == 4:
                obj_rect = self._draw_player(self.size[0]+1, 3, 0, is_help=True)
                self._draw_orb(self.size[0]+1, 3)
            elif i == 5:
                obj_rect = self._draw_player(self.size[0]+1, 4, 0, is_help=True)
                self._draw_pit(self.size[0]+1, 4)
            elif i == 6:
                obj_rect = self._draw_pit(self.size[0]+2, 0)
                self._draw_orb(self.size[0]+2, 0, half_size=True, center=True)
            elif i == 7:
                obj_rect = self._draw_player(self.size[0]+2, 1, 0, is_help=True)
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

    def _compute_reward(self, flag):
        reward = self._reward_function(flag=flag)
        self._current_reward += reward

    @staticmethod
    def _reward_function0(flag):
        match flag:
            case "turned right":
                reward = 0.
            case "tried to move forward and moved forward":
                reward = 0.
            case "tried to move forward but stayed":
                reward = 0.
            case "tried to move forward":
                reward = 0.
            case "tried to move forward to cell type 1":
                reward = 0.
            case "tried to move forward to cell type 2":
                reward = 0.
            case "tried to pick orb up with already having another orb":
                reward = 0.
            case "tried to pick orb up in cell type 4":
                reward = 0.
            case "tried to pick orb up in cell types other than 4":
                reward = 0.
            case "tried to put orb down without having an orb":
                reward = 0.
            case "tried to put orb down on cell type 1":
                reward = 0.
            case "tried to put orb down on cell type 4":
                reward = 0.
            case "tried to put orb down on cell type 5":
                reward = 1.
            case "tried to put orb down on cell type 7":
                reward = 0.
            case _:
                print("Wrong flag for computing reward.")
                raise
        
        return reward

    @staticmethod
    def _reward_function1(flag):
        match flag:
            case "turned right":
                reward = -0.1
            case "tried to move forward and moved forward":
                reward = 0.
            case "tried to move forward but stayed":
                reward = 0.
            case "tried to move forward":
                reward = -0.1
            case "tried to move forward to cell type 1":
                reward = 0 # -0.1
            case "tried to move forward to cell type 2":
                reward = 0 # 0.1
            case "tried to pick orb up with already having another orb":
                reward = -0.1
            case "tried to pick orb up in cell type 4":
                reward = 0. # 0.1
            case "tried to pick orb up in cell types other than 4":
                reward = -0.1
            case "tried to put orb down without having an orb":
                reward = -0.1
            case "tried to put orb down on cell type 1":
                reward = -0.1
            case "tried to put orb down on cell type 4":
                reward = -0.1
            case "tried to put orb down on cell type 5":
                reward = 1.
            case "tried to put orb down on cell type 7":
                reward = -0.1
            case _:
                print("Wrong flag for computing reward.")
                raise

        return reward

    def _do_action(self, action):
        match action:
            case 0:
                self._turn_right()
            case 1:
                self._move_forward()
            case 2:
                self._pick_orb_up()
            case 3:
                self._put_orb_down()
            case _: # not-valid action
                print("Not a valid action!")
                raise

    def _turn_right(self):
        self.players_direction[self.player_turn] = \
            (self.players_direction[self.player_turn] + 1) % len(PitsAndOrbs.DIRECTIONS)
        
        self._compute_reward(flag="turned right")

    def _move_forward(self):
        player_pos_prev = self.players_pos[self.player_turn]
        player_pos_i, player_pos_j = player_pos_prev

        player_cell_type_prev = self.board_state[player_pos_i, player_pos_j]
        match player_cell_type_prev:
            case 1:
                self.board_state[player_pos_i, player_pos_j] = 0
            case 4:
                self.board_state[player_pos_i, player_pos_j] = 2
            case 5:
                self.board_state[player_pos_i, player_pos_j] = 3
            case 7:
                self.board_state[player_pos_i, player_pos_j] = 6

        match self.players_direction[self.player_turn]:
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
            case 1: # two players cannot be in the same cell
                self.board_state[player_pos_prev] = player_cell_type_prev
                player_pos_i, player_pos_j = player_pos_prev

                self._compute_reward(flag="tried to move forward to cell type 1")
            case 2:
                self.board_state[player_pos_i, player_pos_j] = 4

                self._compute_reward(flag="tried to move forward to cell type 2")
            case 3:
                self.board_state[player_pos_i, player_pos_j] = 5
            case 4:
                self.board_state[player_pos_prev] = player_cell_type_prev
                player_pos_i, player_pos_j = player_pos_prev
            case 5:
                self.board_state[player_pos_prev] = player_cell_type_prev
                player_pos_i, player_pos_j = player_pos_prev
            case 6:
                self.board_state[player_pos_i, player_pos_j] = 7
            case 7:
                self.board_state[player_pos_prev] = player_cell_type_prev
                player_pos_i, player_pos_j = player_pos_prev

        if (player_pos_i, player_pos_j) != player_pos_prev: # player actually moved
            self.players_pos[self.player_turn] = (player_pos_i, player_pos_j)
            self.players_movements[self.player_turn] += 1

            self._compute_reward(flag="tried to move forward and moved forward")
        else: # player couldn't move
            self._compute_reward(flag="tried to move forward but stayed")

        self._compute_reward(flag="tried to move forward")

    def _pick_orb_up(self):
        if self.players_have_orb[self.player_turn]:
            self._compute_reward(flag="tried to pick orb up with already having another orb")

            return

        player_pos_i, player_pos_j = self.players_pos[self.player_turn]
        if self.board_state[player_pos_i, player_pos_j] == 4:
            self.board_state[player_pos_i, player_pos_j] = 1
            self.players_have_orb[self.player_turn] = True

            self._compute_reward(flag="tried to pick orb up in cell type 4")
        else:
            self._compute_reward(flag="tried to pick orb up in cell types other than 4")

    def _put_orb_down(self):
        if not self.players_have_orb[self.player_turn]:
            self._compute_reward(flag="tried to put orb down without having an orb")

            return

        player_pos_i, player_pos_j = self.players_pos[self.player_turn]

        match self.board_state[player_pos_i, player_pos_j]:
            case 1:
                self.board_state[player_pos_i, player_pos_j] = 4
                self.players_have_orb[self.player_turn] = False

                self._compute_reward(flag="tried to put orb down on cell type 1")
            case 4:
                self._compute_reward(flag="tried to put orb down on cell type 4")
            case 5:
                self.board_state[player_pos_i, player_pos_j] = 7
                self.players_have_orb[self.player_turn] = False

                self._move_orbs_randomly()

                self._compute_reward(flag="tried to put orb down on cell type 5")
            case 7:
                self._compute_reward(flag="tried to put orb down on cell type 7")

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

    def _calc_filled_pits(self):
        cell_type6_num = len(np.where(self.board_state==6)[0])
        cell_type7_num = len(np.where(self.board_state==7)[0])

        return cell_type6_num+cell_type7_num

    def _change_player_turn(self):
        self.player_turn = (self.player_turn + 1) % self.players_num

    def play1(self, show_obs_or_state=0, show_help=True, 
                clear=True): # input=4 quits the game
        rewards = 0

        while True:
            self.show_board(show_obs_or_state=show_obs_or_state, 
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
                print(f"Game ended successfully with {self.players_movements} movements.")
                break

    def play2(self, show_obs_or_state=0, show_help=False, 
            clear=False, print_is_enabled=True): # play function for pygame
        obs = None
        rewards = 0
        done = False
        info = None
        if print_is_enabled:
            self.show_board(obs=obs, info=info, 
                            show_obs_or_state=show_obs_or_state, 
                            show_help=show_help, clear=clear)

        while True:
            self._update_screen()

            action = self._check_events()

            if action is not None and not done:
                obs, reward, done, info = self.step(action)
                rewards += reward

                if print_is_enabled:
                    self.show_board(obs=obs, info=info, 
                                    show_obs_or_state=show_obs_or_state, 
                                    show_help=show_help, clear=clear)

                    print("Taken reward for last action:", reward)
                    print("All rewards until now:", rewards)
                    print()

                    if done and not self.printed_game_is_finished:
                        print(f"Game ended successfully with {self.players_movements} movements.")
                        self._build_finish_materials()
                        self.printed_game_is_finished = True

            self.clock.tick(PitsAndOrbs.FPS)

    def step(self, action):
        self._do_action(action)

        obs = self.get_obs()
        reward = self._current_reward
        done = self.is_done()
        info = self.get_info()        

        self._change_player_turn()

        # reset current step's rewards
        self._current_reward = 0

        return obs, reward, done, info

    def reset_game(self, seed=None):
        # initiating variables
        self.printed_game_is_finished = False
        self._current_reward = 0.

        seed = self.seed if seed is None else seed
        np.random.seed(seed)

        self.board_state = np.zeros(self.size, dtype=np.uint8)
        indices = list(range(self.board_state.size))

        random_cells = np.random.choice(indices, size=self.pit_num+self.orb_num+self.players_num, replace=False)
        pit_indices = random_cells[: self.pit_num]
        orb_indices = random_cells[self.pit_num: self.pit_num+self.orb_num]
        players_indices = random_cells[-self.players_num:]

        self.board_state = self.board_state.flatten()

        # put the player(s) randomly in the board
        self.board_state[players_indices] = 1

        # create random pits and orbs in the board
        self.board_state[orb_indices] = 2
        self.board_state[pit_indices] = 3
        
        self.board_state = self.board_state.reshape(self.size)

        # initiating multiple players
        self.player_turn = 0 # (0, ..., self.player_num-1)
        self.players_pos = [(player_index // self.size[1], player_index % self.size[1]) \
                            for player_index in players_indices]
        self.players_direction = [np.random.randint(len(PitsAndOrbs.DIRECTIONS)) \
                            for _ in range(self.players_num)]
        self.players_have_orb = [False for _ in range(self.players_num)]
        self.players_movements = [0 for _ in range(self.players_num)]

        # updating shared memory according to every players' neighbors
        self.memory = Memory(self)
        for i in range(1, self.players_num): 
            self.player_turn = i
            self.get_obs()
        self.player_turn = 0

        # initiating game-runner functions
        if self._pygame_mode:
            pygame.init()

            self.multiplier = 100
            self.border_color = (255, 120, 0)
            self.border_width = 3
            self.border_margin = self.multiplier - (1.5 * self.border_width)

            self.play = self.play2

            try:
                icon = pygame.image.load("./game/pao.ico")
            except:
                icon = pygame.image.load("./pao.ico")
            pygame.display.set_icon(icon)

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

            self._update_screen() # render everything on the PyGame window for the first time
        else:
            self.play = self.play1           

        # getting the first observation and information
        obs = self.get_obs()
        info = self.get_info()

        return obs, info

    def get_neighbors(self):
        padded_board_state = np.zeros((self.size[0]+2, self.size[1]+2), dtype=np.uint8)
        padded_board_state[0, :] = len(PitsAndOrbs.CELLS)
        padded_board_state[-1, :] = len(PitsAndOrbs.CELLS)
        padded_board_state[:, 0] = len(PitsAndOrbs.CELLS)
        padded_board_state[:, -1] = len(PitsAndOrbs.CELLS)
        padded_board_state[1:-1, 1:-1] = self.board_state

        player_pos_i, player_pos_j = self.players_pos[self.player_turn]
        player_pos_i += 1
        player_pos_j += 1
        obs = padded_board_state[player_pos_i-1: player_pos_i+2, player_pos_j-1: player_pos_j+2]

        return obs

    def get_obs(self):
        if self._return_obs_type == "partial obs":
            player_pos = self.players_pos[self.player_turn]
            neighbors = self.get_neighbors()

            self.memory.update(neighbors, player_pos)
            partial_obs_with_mem = self.memory.get()

            return partial_obs_with_mem
        elif self._return_obs_type == "full state":
            full_state = self.board_state.copy()

            return full_state
        elif self._return_obs_type == "neighbors":
            partial_obs_with_neighbors = self.get_neighbors()

            return partial_obs_with_neighbors

    def get_frame(self):
        assert self._pygame_mode

        frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

        return frame

    def is_done(self):
        done = (self._calc_filled_pits() == self.orb_num)

        return done

    def get_info(self):
        return {
            "filled pits#": self._calc_filled_pits(),
            "all pits#": self.pit_num,
            **{f"player{i} movements#": self.players_movements[i] for i in range(self.players_num)},
        }

    def clear_screen(self):
        os.system("cls" if os.name=="nt" else "clear")

    def show_board(self, obs=None, info=None, show_obs_or_state=0, 
                    show_help=True, clear=True): # show_obs_or_state=0 prints observation and show_obs_or_state=1 prints state
        obs = obs if obs is not None else self.get_obs()
        info = info if info is not None else self.get_info()

        if clear:
            self.clear_screen()

        if show_obs_or_state == 0:
            print(obs)
            print()
        elif show_obs_or_state == 1:
            print(self.board_state)
            print()

        print(info)
        print()

        if show_help:
            print("Directions:", PitsAndOrbs.DIRECTIONS)
            print("Cell Types:", PitsAndOrbs.CELLS)
            print("Actions:", PitsAndOrbs.ACTIONS)
            print()
       
    def close_game(self):
        if self._pygame_mode:
            self.board_state = None
    
            pygame.quit()


if __name__ == "__main__":
    game = PitsAndOrbs()
    game.reset_game()

    game.play()