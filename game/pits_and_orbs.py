import pygame

import numpy as np

import sys

import os

try:
    from game.team import Team
except:
    from team import Team


class PitsAndOrbs:
    DIRECTIONS = ["0.west", "1.north", "2.east", "3.south"]
    CELLS = ["0.nothing", "1.player", "2.orb", "3.pit", "4.player&orb", 
            "5.player&pit", "6.orb&pit", "7.player&orb&pit"] # , "8.out of bound"
    ACTIONS = ["0.turn right", "1.move forward", "2.pick orb up", 
            "3.put orb down", "4.throw orb away"]

    OBSERVATIONS = ["partial obs", "full state", "neighbors"]
    BOARDS = ["array", "positions"]

    TEAM_COLORS = [(0, 0, 255), (122, 0, 255)]

    FPS = 60

    def __init__(self, 
                size=(5, 5), orb_num=5, pit_num=5, player_num=1, team_num=1, 
                return_obs_type="partial obs", return_board_type="array", 
                pygame_mode=True, pygame_with_help=True, 
                reward_function_type='0', reward_function=None, seed=None):
        if isinstance(size, int):
            self.size = (size, size)
        elif len(size) == 2:
            self.size = size
        else:
            raise Exception("Wrong size for game's grid; expected size argument's shape to be 1 or 2.")

        assert (self.size[0] * self.size[1]) > (orb_num + pit_num + player_num)
        self.orb_num = orb_num
        self.pit_num = pit_num
        self.player_num = player_num

        assert player_num % team_num == 0
        self.team_num = team_num

        self.team_size = self.player_num // self.team_num

        assert return_obs_type.lower() in PitsAndOrbs.OBSERVATIONS
        self._return_obs_type = return_obs_type.lower()

        assert return_board_type.lower() in PitsAndOrbs.BOARDS
        self._return_board_type = return_board_type.lower()

        assert not (self._return_obs_type == "neighbors" and self._return_board_type == "positions")

        self._pygame_mode = pygame_mode
        self._pygame_with_help = pygame_with_help

        if reward_function is None:
            if reward_function_type == '0':
                self._reward_function = PitsAndOrbs._reward_function0
            elif reward_function_type == '1':
                self._reward_function = PitsAndOrbs._reward_function1
        else:
            self._reward_function = reward_function

        self.seed = seed

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
                    case pygame.K_4:
                        action = 4
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

    def _display_objects(self): # TODO: highlight neighbor cells + mark filled pits with their corresponding team color + create seperate windows for each team
        if self._return_obs_type in ("partial obs", "neighbors"):
            boards = [team.get_memory() for team in self.teams]
        elif self._return_obs_type == "full state":
            boards = [self.board_state for _ in self.teams]

        for team_index, (board, team , team_color) in enumerate(zip(boards, self.teams, PitsAndOrbs.TEAM_COLORS)):
            for player_index, (player_pos_i, player_pos_j) in enumerate(team.players_pos):
                self._draw_player(player_pos_i, player_pos_j, team_index, player_index, team_color)

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

        for team_index, team in enumerate(self.teams):
            for player_index, (player_pos_i, player_pos_j) in enumerate(team.players_pos):
                self._draw_player_direction(player_pos_i, player_pos_j, team_index, player_index)

    def _draw_player(self, i, j, team_index, player_index, player_color, is_help=False):
        rect = pygame.draw.rect(
            self.screen, 
            player_color, 
            (j*self.multiplier+self.border_width, 
            i*self.multiplier+self.border_width, 
            self.border_margin, self.border_margin)
        )

        if not is_help:
            player_index_text = self.smaller_font.render(str(player_index), True, (0, 0, 0), (200, 200, 200))
            player_index_rect = player_index_text.get_rect()
            player_index_rect.topright = rect.topright
            self.screen.blit(player_index_text, player_index_rect)

        if self.teams[team_index].players[player_index].has_orb and not is_help:
            self._draw_orb(i, j, half_size=True)

        return rect

    def _draw_player_direction(self, i, j, team_index, player_index):
        match self.teams[team_index].players[player_index].direction:
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
        for i, team in enumerate(self.teams):
            for j, player in enumerate(team.players):
                player_turn_text = ""
                if self.team_turn == i and team._player_turn == j:
                    player_turn_text = "has to play now!"

                player_movement_txt = self.smaller_font.render(f"Team{i} Scores: #{team.scores}"+\
                                    " || "+f"Player{j} Movements: #{player.movements}"+" || "+\
                                    player_turn_text.upper(), True, (0, 0, 0), (200, 200, 200))
                player_movement_rect = player_movement_txt.get_rect()
                player_movement_rect.topleft = (0, self.size[0]*self.multiplier+self.border_width+10+15*(i*2+j))
                self.screen.blit(player_movement_txt, player_movement_rect)

        help_txt = self.smaller_font.render(
            f"{'    |    '.join([str(i)+' ==> '+action.split('.')[-1].upper() for i, action in enumerate(PitsAndOrbs.ACTIONS[:-1])])}", 
            True, 
            (0, 0, 0), 
            (200, 200, 200)
        )
        help_rect = help_txt.get_rect()
        help_rect.topleft = (player_movement_rect.bottomleft[0]+self.border_width, 
                            player_movement_rect.bottomleft[1]+self.multiplier/10)
        self.screen.blit(help_txt, help_rect)
        
        if self.team_num > 1:
            help_txt_last = self.smaller_font.render(
                '4 ==> ' + PitsAndOrbs.ACTIONS[-1].split('.')[-1].upper() + '    |', 
                True, 
                (0, 0, 0), 
                (200, 200, 200)
            )
            help_rect_last = help_txt_last.get_rect()
            help_rect_last.topleft = (help_rect.bottomleft[0], 
                                    help_rect.bottomleft[1]+self.multiplier/30)
            self.screen.blit(help_txt_last, help_rect_last)

        for i, cell_type in enumerate(PitsAndOrbs.CELLS):
            type_ = cell_type.split('.')[-1]
            if i == 0:
                continue
            elif i == 1:
                obj_rect = self._draw_player(self.size[0]+1, 0, 0, 0, PitsAndOrbs.TEAM_COLORS[0], is_help=True)
            elif i == 2:
                obj_rect = self._draw_orb(self.size[0]+1, 1)
            elif i == 3:
                obj_rect = self._draw_pit(self.size[0]+1, 2)
            elif i == 4:
                obj_rect = self._draw_player(self.size[0]+1, 3, 0, 0, PitsAndOrbs.TEAM_COLORS[0], is_help=True)
                self._draw_orb(self.size[0]+1, 3)
            elif i == 5:
                obj_rect = self._draw_player(self.size[0]+1, 4, 0, 0, PitsAndOrbs.TEAM_COLORS[0], is_help=True)
                self._draw_pit(self.size[0]+1, 4)
            elif i == 6:
                obj_rect = self._draw_pit(self.size[0]+2, 0)
                self._draw_orb(self.size[0]+2, 0, half_size=True, center=True)
            elif i == 7:
                obj_rect = self._draw_player(self.size[0]+2, 1, 0, 0, PitsAndOrbs.TEAM_COLORS[0], is_help=True)
                self._draw_pit(self.size[0]+2, 1)
                self._draw_orb(self.size[0]+2, 1, half_size=True, center=True)

            team_text = "team0 " if type_ == "player" else ""
            self._write_text(team_text+type_, obj_rect)

        if self.team_num > 1:
            obj_rect = self._draw_player(self.size[0]+2, 2, 0, 0, PitsAndOrbs.TEAM_COLORS[1], is_help=True)
            self._write_text("team1 "+"player", obj_rect)

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

    def _add_to_current_reward(self, flag):
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
            case "tried to throw another team's orb away in cell type 7":
                reward = 1.
            case "tried to throw its team's orb away in cell type 7":
                reward = 0.
            case "tried to throw orb away in cell types other than 7":
                reward = 0.
            case "the player has depleted its movements":
                reward = 0.
            case "episode is done successfully":
                reward = 1.
            case "episode is done unsuccessfully":
                reward = 0.
            case "episode is not done":
                reward = 0.
            case _:
                raise Exception("Wrong flag for computing reward.")
        
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
                reward = 0. # -0.1
            case "tried to move forward to cell type 2":
                reward = 0. # 0.1
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
            case "tried to throw another team's orb away in cell type 7":
                reward = 1.
            case "tried to throw its team's orb away in cell type 7":
                reward = -1.
            case "tried to throw orb away in cell types other than 7":
                reward = -0.1
            case "the player has depleted its movements":
                reward = -0.1
            case "episode is done successfully":
                reward = 1.
            case "episode is done unsuccessfully":
                reward = -1.
            case "episode is not done":
                reward = 0.
            case _:
                raise Exception("Wrong flag for computing reward.")

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
            case 4:
                if self.team_num != 1:
                    self._throw_orb_away()
                else:
                    raise Exception("The action 'throw orb away' isn't available for one-team game mode.")
            case _: # not-valid action
                raise Exception("Not a valid action!")

    def _turn_right(self):
        self.current_player.direction = \
            (self.current_player.direction + 1) % len(PitsAndOrbs.DIRECTIONS)

        self._add_to_current_reward(flag="turned right")

    def _move_forward(self):
        player_pos_prev = self.current_player.position
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

        match self.current_player.direction:
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

                self._add_to_current_reward(flag="tried to move forward to cell type 1")
            case 2:
                self.board_state[player_pos_i, player_pos_j] = 4

                self._add_to_current_reward(flag="tried to move forward to cell type 2")
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
            self.current_player.position = (player_pos_i, player_pos_j)
            self.current_player.movements += 1

            self._add_to_current_reward(flag="tried to move forward and moved forward")
        else: # player couldn't move
            self._add_to_current_reward(flag="tried to move forward but stayed")

        self._add_to_current_reward(flag="tried to move forward")

    def _pick_orb_up(self):
        if self.current_player.has_orb:
            self._add_to_current_reward(flag="tried to pick orb up with already having another orb")
            return

        player_pos_i, player_pos_j = self.current_player.position
        if self.board_state[player_pos_i, player_pos_j] == 4:
            self.board_state[player_pos_i, player_pos_j] = 1
            try:
                self.current_player.add_orb(
                    orb_index=self._orbs_pos.index((player_pos_i, player_pos_j)))
            except ValueError:
                self._write_status_to_file()
                raise

            self._add_to_current_reward(flag="tried to pick orb up in cell type 4")
        else:
            self._add_to_current_reward(flag="tried to pick orb up in cell types other than 4")

    def _put_orb_down(self):
        if not self.current_player.has_orb:
            self._add_to_current_reward(flag="tried to put orb down without having an orb")

            return

        player_pos_i, player_pos_j = self.current_player.position

        match self.board_state[player_pos_i, player_pos_j]:
            case 1:
                self.board_state[player_pos_i, player_pos_j] = 4
                self._orbs_pos[self.current_player.remove_orb()] \
                    = (player_pos_i, player_pos_j)

                self._add_to_current_reward(flag="tried to put orb down on cell type 1")
            case 4:
                self._add_to_current_reward(flag="tried to put orb down on cell type 4")
            case 5:
                self.board_state[player_pos_i, player_pos_j] = 7
                self._orbs_pos[self.current_player.remove_orb()] \
                    = (player_pos_i, player_pos_j)

                self.current_team.add_to_filled_pits((player_pos_i, player_pos_j))
                self._move_orbs_randomly()

                self._add_to_current_reward(flag="tried to put orb down on cell type 5")
            case 7:
                self._add_to_current_reward(flag="tried to put orb down on cell type 7")

    def _throw_orb_away(self):
        player_pos_i, player_pos_j = self.current_player.position
        if self.board_state[player_pos_i, player_pos_j] == 7:
            self.board_state[player_pos_i, player_pos_j] = 5
            self._throw_orb_randomly(orb_pos=(player_pos_i, player_pos_j))

            for team in self.teams:
                if (player_pos_i, player_pos_j) in team.filled_pits:
                    team.rem_from_filled_pits((player_pos_i, player_pos_j))

                    self._add_to_current_reward(flag="tried to throw another team's orb away in cell type 7")
                    break
            else:
                self._add_to_current_reward(flag="tried to throw its team's orb away in cell type 7")
            
        else:
            self._add_to_current_reward(flag="tried to throw orb away in cell types other than 7")

    def _throw_orb_randomly(self, orb_pos):
        orb_index = self._orbs_pos.index(orb_pos)

        valid_new_orb_poses = np.where(self.board_state == 0) # only empty cells are valid for orbs to be thrown to
        if len(valid_new_orb_poses) > 0:
            valid_new_orb_poses = list(zip(*valid_new_orb_poses))

            new_orb_pos_index = np.random.randint(len(valid_new_orb_poses)-1)
            new_orb_pos = valid_new_orb_poses[new_orb_pos_index]

            self.board_state[new_orb_pos] = 2
            self._orbs_pos[orb_index] = new_orb_pos
        else:
            pass # there are no empty cells left

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
                    self._move_orb_forward(orb_pos, direction)

    def _move_orb_forward(self, orb_pos, direction):
        orb_index = self._orbs_pos.index(orb_pos)
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
                self._orbs_pos[orb_index] = (orb_pos_i, orb_pos_j)
            case 1:
                self.board_state[orb_pos_i, orb_pos_j] = 4
                self._orbs_pos[orb_index] = (orb_pos_i, orb_pos_j)
            case 2:
                self.board_state[orb_pos] = prev_cell
            case 3:
                self.board_state[orb_pos_i, orb_pos_j] = 6
                self._orbs_pos[orb_index] = (orb_pos_i, orb_pos_j)
            case 4:
                self.board_state[orb_pos] = prev_cell
            case 5:
                self.board_state[orb_pos_i, orb_pos_j] = 7
                self._orbs_pos[orb_index] = (orb_pos_i, orb_pos_j)
            case 6:
                self.board_state[orb_pos] = prev_cell

    def _write_status_to_file(self):
        if not os.path.exists("./debug"):
            os.makedirs("./debug")

        np.savetxt("./debug/board.txt", self.board_state, delimiter=",")

        np.savetxt("./debug/memory.txt", self.current_team.get_memory(), delimiter=",")

        with open("./debug/orbs_pos.txt", "w") as file:
            for orb_pos in self._orbs_pos:
                file.write(str(orb_pos) + "\n")

        with open("./debug/current_player_position.txt", "w") as file:
            file.write(str(self.current_player.position) + "\n")

    def _calc_filled_pits(self):
        cell_type6_num = len(np.where(self.board_state==6)[0])
        cell_type7_num = len(np.where(self.board_state==7)[0])

        return cell_type6_num+cell_type7_num

    def _change_player_and_team_turn(self):
        change_team = self.current_team.change_player_turn()

        if change_team:
            self.team_turn = (self.team_turn + 1) % self.team_num

    def _change_team_and_player_turn(self):
        self.current_team.change_player_turn()
        
        self.team_turn = (self.team_turn + 1) % self.team_num        

    def _is_done(self):
        done = (self._calc_filled_pits() == self.orb_num)

        return done

    def _get_observation(self):
        if self._return_obs_type == "partial obs":
            neighbors = self.current_team.get_neighbors()
            self.current_team.update_memory(neighbors)
            partial_obs_with_mem = self.current_team.get_memory()

            obs = partial_obs_with_mem
        elif self._return_obs_type == "full state":
            full_state = self.board_state.copy()
    
            obs =  full_state
        elif self._return_obs_type == "neighbors":
            partial_obs_with_neighbors = self.current_team.get_neighbors()

            obs = partial_obs_with_neighbors
    
        if self._return_board_type == "positions":
            obs = self._convert_array_to_positions(obs)

        return obs
    
    def _get_info(self):
        return {
            "all_pits#": self.pit_num,
            "total_filled_pits#": self._calc_filled_pits(),
            **{f"team{i}_filled_pits#": self.teams[i].scores for i in range(self.team_num)},
            **{f"team{i}_player{j}_movements#": self.teams[i].players[j].movements for i in range(self.team_num) for j in range(self.team_size)},
        }

    def _convert_array_to_positions(self, array=None):
        if self._return_obs_type == "partial obs":
            assert array is not None

        poses = []
        if self._return_obs_type == "full state":
            poses += self._pits_pos
            poses += self._orbs_pos
        elif self._return_obs_type == "partial obs":
            not_seen_index_i, not_seen_index_j = self.size

            pits_index = []
            for pit_cell in (3, 5, 6, 7):
                pit_pos_Is, pit_pos_Js = np.where(array==pit_cell)
                if len(pit_pos_Is) < 1:
                    continue

                for pit_pos in zip(pit_pos_Is, pit_pos_Js):
                    pits_index.append(self._pits_pos.index(pit_pos))
            pits_index.sort()
            for pit_index in range(self.pit_num):
                if pit_index in pits_index:
                    poses.append(self._pits_pos[pit_index])
                else:
                    poses.append((not_seen_index_i, not_seen_index_j))

            orbs_index = []
            for orb_cell in (2, 4, 6, 7):
                orb_pos_Is, orb_pos_Js = np.where(array==orb_cell)
                if len(orb_pos_Is) < 1:
                    continue

                for orb_pos in zip(orb_pos_Is, orb_pos_Js):
                    try:
                        orbs_index.append(self._orbs_pos.index(orb_pos))
                    except ValueError:
                        pass # This happens due to inconsistensy between board and memory since an orb randomly has moved and it has not yet registered in the memory.
                    
            orbs_index.sort()
            for orb_index in range(self.orb_num):
                if orb_index in orbs_index:
                    poses.append(self._orbs_pos[orb_index])
                else:
                    poses.append((not_seen_index_i, not_seen_index_j))

        return np.array(poses)

    @property
    def current_team(self):
        return self.teams[self.team_turn]

    @property
    def current_player(self):
        return self.current_team.current_player

    def step_game(self, action):
        self._do_action(action)

        board = self._get_observation()
        reward = self._current_reward
        done = self._is_done()
        info = self._get_info()

        self._change_team_and_player_turn()

        # reset current step's rewards
        self._current_reward = 0.

        return board, reward, done, info

    def reset_game(self, seed=None):
        # initiate necessary variables
        self.printed_game_is_finished = False
        self._current_reward = 0.

        seed = self.seed if seed is None else seed
        np.random.seed(seed)

        self.board_state = np.zeros(self.size, dtype=np.uint8)
        indices = list(range(self.board_state.size))

        random_cells = np.random.choice(indices, size=self.pit_num+self.orb_num+self.player_num, replace=False)
        pit_indices = random_cells[: self.pit_num]
        orb_indices = random_cells[self.pit_num: self.pit_num+self.orb_num]
        players_indices = random_cells[-self.player_num:]

        self.board_state = self.board_state.flatten()

        # put the player(s) randomly in the board
        self.board_state[players_indices] = 1

        # create random orbs and pits in the board
        self.board_state[orb_indices] = 2
        self.board_state[pit_indices] = 3
        
        self.board_state = self.board_state.reshape(self.size)

        # save orbs and pits positions and their orders
        if self._return_board_type == "positions":
            orb_pos_Is, orb_pos_Js = np.where(self.board_state==2)
            self._orbs_pos = list(zip(orb_pos_Is, orb_pos_Js))

            pit_pos_Is, pit_pos_Js = np.where(self.board_state==3)
            self._pits_pos = list(zip(pit_pos_Is, pit_pos_Js))

        # initiate multiple teams with multiple players
        self.team_turn = 0
        self.teams = [Team(game=self, players_index=players_indices[i*self.team_size: (i+1)*self.team_size])\
                    for i in range(self.team_num)]

        # initiate game-runner functions
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

        # get the first observation and information
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def render_game(self, obs=None, info=None,  
                    show_help=True, clear=True):
        obs = obs if obs is not None else self._get_observation()
        info = info if info is not None else self._get_info()

        if clear:
            PitsAndOrbs.clear_screen()

        print(obs)
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

    def get_frame(self):
        assert self._pygame_mode

        frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

        return frame

    @staticmethod
    def clear_screen():
        os.system("cls" if os.name=="nt" else "clear")

    def play1(self, show_help=True, clear=True): # input=4 quits the game
        rewards = 0.

        while True:
            self.render_game(show_help=show_help, clear=clear)

            action = int(input("Next Action: "))
            if action == 4:
                print("Quiting the game ...")
                break

            _, reward, done, _ = self.step_game(action)
            rewards += reward

            print("Taken reward for last action:", reward)
            print("All rewards until now:", rewards)
            print()

            if done:
                print(f"Game ended successfully.")
                print("Final info:", info)
                break

    def play2(self, show_help=False, clear=False, print_is_enabled=True): # play function for pygame
        obs = None
        rewards = 0.
        done = False
        info = None
        if print_is_enabled:
            self.render_game(obs=obs, info=info, show_help=show_help, clear=clear)

        while True:
            self._update_screen()

            action = self._check_events()

            if action is not None and not done:
                obs, reward, done, info = self.step_game(action)
                rewards += reward

                if print_is_enabled:
                    self.render_game(obs=obs, info=info, show_help=show_help, clear=clear)

                    print("Taken reward for last action:", reward)
                    print("All rewards until now:", rewards)
                    print()

                    if done and not self.printed_game_is_finished:
                        print(f"Game ended successfully.")
                        print("Final info:", info)
                        self._build_finish_materials()
                        self.printed_game_is_finished = True

            self.clock.tick(PitsAndOrbs.FPS)


if __name__ == "__main__":
    game = PitsAndOrbs()
    game.reset_game()

    game.play()