import numpy as np
import random
import matplotlib.pyplot as plt

class Gridworld:
    """
    This env is a gridworld. The agent must go from a start to a stop.
    The wind direction is a tuple : (0, -1) for wind to the left
    The world is a numpy array with :
        7 for the start positions
        8 for the end position
        9 for walls
        0-5 for the wind intensity
    The agent can go in all 9 direction (-1,-1) to (1,1)
    """
    def __init__(self, world, wind_dir):
        self.wind_dir = wind_dir
        self._set_world(world)

    def _set_world(self, world):
        self.world = np.flipud(world)

        self.starts = np.transpose(np.where(world == 7))
        self.starts = [(x[0], x[1]) for x in self.starts]
        self.position = random.choice(self.starts)

        self.ends = np.transpose(np.where(world == 8))
        self.ends = [(x[0], x[1]) for x in self.ends]
        self.end = random.choice(self.ends)

        self.world[world == 7] = 0
        self.world[world == 8] = 0
        self.walls = np.transpose(np.where(world == 9))
        self.walls = [(x[0], x[1]) for x in self.walls]

    def reset(self):
        self.position = random.choice(self.starts)
        self.position_memory = [self.position]
        return self.position

    def _is_wall(self, pos):
        return pos in self.walls

    def valid_actions(self):
        actions = []
        for x,y in np.ndindex((3,3)):
            actions.append((x - 1, y - 1))
        # actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        return actions

    def valid_states(self):
        return [p for p in np.ndindex(np.shape(self.world))]

    def get_state(self):
        return self.position

    def _move(self, action):
        pos = np.array(self.position) + np.array(action)
        pos = np.clip(pos, (0, 0), np.array(np.shape(self.world)) - 1)
        pos = tuple(pos)

        if self._is_wall(pos):
            return self.position
        else:
            return pos

    def step(self, action):
        done = False
        reward = -1

        self.position = self._move(action)
        wind = self.world[self.position[0], self.position[1]]
        wind += np.random.choice(3) - 1
        wind = max(0, wind)
        for i in range(wind):
            self.position = self._move(self.wind_dir)

        if self.position == self.end:
            done = True

        self.position_memory.append(self.position)

        return self.position, reward, done

    def render(self):
        state = np.flipud(self.world.copy())
        for p in self.position_memory:
            state[p[0], p[1]] = 4
        state[self.end[0], self.end[1]] = 8

        plt.imshow(state, cmap='hot')
        plt.show()

world = np.array([
    [0, 0, 0 , 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0 , 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0 , 9, 9, 1, 2, 2, 1, 0],
    [7, 0, 0 , 9, 9, 1, 2, 8, 1, 0],
    [0, 0, 0 , 9, 9, 1, 2, 2, 1, 0],
    [0, 0, 0 , 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0 , 1, 1, 1, 2, 2, 1, 0]
])


env = Gridworld(world, (1, 0))