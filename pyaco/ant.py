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


@numba.jit(nopython=True)
def weighted_random_choice(choices, probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    random_choice = np.random.rand()
    for i, prob in enumerate(cumulative_probabilities):
        if random_choice < prob:
            return choices[i]
    return choices[-1]


ANT_MOVES = np.array(
    [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=np.int64,
)


@numba.jit(nopython=True)
def _observe(ant, grid: np.ndarray, occupied_squares: np.ndarray):
    local_grid = grid[max(0, ant.y - 3) : ant.y + 2, max(0, ant.x - 3) : ant.x + 2]
    local_occupied = occupied_squares[
        max(0, ant.y - 3) : ant.y + 2, max(0, ant.x - 3) : ant.x + 2
    ]

    pheromone_values = np.empty(len(ANT_MOVES), dtype=np.float64)

    center_y, center_x = 1, 1  # Center of the 3x3 grid

    prev_dy, prev_dx = 0, 0
    if ant.last_x != -1 and ant.last_y != -1:
        prev_dy = ant.y - ant.last_y
        prev_dx = ant.x - ant.last_x

    for idx, (dy, dx) in enumerate(ANT_MOVES):
        grid_y, grid_x = center_y + dy, center_x + dx

        if 0 <= grid_y < 3 and 0 <= grid_x < 3:
            pheromone_value = local_grid[grid_y, grid_x]

            # Check if square is occupied
            if local_occupied[grid_y, grid_x] == 1:  # Ant present
                pheromone_value = -5.0
            elif local_occupied[grid_y, grid_x] == 2:  # Food present
                pheromone_value = 50.0

            # Bias towards forward motion
            if dy == prev_dy and dx == prev_dx:
                pheromone_value *= 1.5

            pheromone_values[idx] = pheromone_value
        else:
            if dy == prev_dy and dx == prev_dx:
                pheromone_values[idx] = -5.0
            else:
                pheromone_values[idx] = -10.0

    probabilities = np.exp(pheromone_values)
    probabilities /= np.sum(probabilities)
    action = weighted_random_choice(np.arange(8), probabilities)
    return action


@numba.experimental.jitclass(ant_spec)
class Ant:
    def __init__(self, x, y, color=(255, 0, 0), pheromone_amount=0.01):
        self.color = np.array(color, dtype=np.int32)

        self.x = x
        self.y = y
        self.last_x = -1  # Use -1 to indicate no previous position
        self.last_y = -1
        self.second_last_x = -1
        self.second_last_y = -1

        self.pheromone_amount = pheromone_amount

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

    def observe(self, grid, occupied_squares):
        return _observe(self, grid, occupied_squares)
