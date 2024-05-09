import numpy as np


class Player:

    def __init__(self, game, player_index):
        self.game = game

        self._position = (player_index // game.size[1], player_index % game.size[1])
        self._direction = np.random.randint(len(game.DIRECTIONS))
        self._has_orb = False
        self._movements = 0

    @property
    def position(self):
        return self._position
    
    @property
    def direction(self):
        return self._direction

    @property
    def has_orb(self):
        return self._has_orb

    @property
    def movements(self):
        return self._movements

    @position.setter
    def position(self, value):
        assert len(value) == 2
        assert value[0] < self.game.size[0] 
        assert value[1] < self.game.size[1]

        self._position = value
    
    @direction.setter
    def direction(self, value):
        assert value >= 0 and value < 4

        self._direction = value

    @has_orb.setter
    def has_orb(self, value):
        assert isinstance(value, bool)

        self._has_orb = value

    @movements.setter
    def movements(self, value):
        assert value >= 0

        self._movements = value