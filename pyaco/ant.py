import numpy as np
import numba
from numba import int32, int64, float64, boolean

ant_spec = [
    ("x", int64),
    ("y", int64),
    ("last_x", int64),
    ("last_y", int64),
    ("second_last_x", int64),
    ("second_last_y", int64),
    ("color", int32[:]),
    ("pheromone_amount", float64),
]


@numba.experimental.jitclass(ant_spec)
class Ant:
    def __init__(self, x, y, color=(255, 0, 0)):
        self.color = np.array(color, dtype=np.int32)

        self.x = x
        self.y = y
        self.last_x = -1  # Use -1 to indicate no previous position
        self.last_y = -1
        self.second_last_x = -1
        self.second_last_y = -1

        self.pheromone_amount = 0.01

    def move(self, action):
        if action == 0:  # Move up
            self.y += 1
        elif action == 1:  # Move right
            self.x += 1
        elif action == 2:  # Move down
            self.y -= 1
        elif action == 3:  # Move left
            self.x -= 1
        elif action == 4:  # Move up-right
            self.y += 1
            self.x += 1
        elif action == 5:  # Move down-right
            self.y -= 1
            self.x += 1
        elif action == 6:  # Move down-left
            self.y -= 1
            self.x -= 1
        elif action == 7:  # Move up-left
            self.y += 1
            self.x -= 1

    def choose_action(self):
        # For now, just return a random action
        return np.random.randint(0, 8)
