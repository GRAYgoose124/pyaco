import numpy as np
import numba
from numba import int32, int64, float64, boolean

ant_spec = [
    ("x", int64),
    ("y", int64),
    ("last_xys", int64[:, :]),
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
    # average all lastxys to get a direction vector
    for last_xy in ant.last_xys:
        if last_xy[0] != -1 and last_xy[1] != -1:
            prev_dy += last_xy[0]
            prev_dx += last_xy[1]
    prev_dy /= len(ant.last_xys)
    norm_prev = np.sqrt(prev_dy**2 + prev_dx**2)

    for idx, (dy, dx) in enumerate(ANT_MOVES):
        grid_y, grid_x = (center_y + dy) % grid.shape[0], (center_x + dx) % grid.shape[
            1
        ]
        if grid_y != (center_y + dy):
            dy = (
                (grid_y - center_y)
                if (grid_y - center_y) != 0
                else -np.sign(dy) * (grid.shape[0] - 1)
            )
        if grid_x != (center_x + dx):
            dx = (
                (grid_x - center_x)
                if (grid_x - center_x) != 0
                else -np.sign(dx) * (grid.shape[1] - 1)
            )

        pheromone_value = local_grid[grid_y, grid_x]

        # Check if square is occupied
        if local_occupied[grid_y, grid_x] == 1:  # Ant present
            pheromone_value *= 0.9
        elif local_occupied[grid_y, grid_x] == 2:  # Food present
            pheromone_value *= 2.0

        # Bias towards forward motion
        norm_current = np.sqrt(dy**2 + dx**2)
        if norm_prev != 0 and norm_current != 0:
            cos_angle = (dy * prev_dy + dx * prev_dx) / (norm_prev * norm_current)
            bias_factor = (1.0 + cos_angle) * 0.1
            pheromone_value *= bias_factor

        pheromone_values[idx] = pheromone_value

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
        #     ("last_xys", List(Tuple([int64, int64]))),
        self.last_xys = np.array([(-1, -1)] * 10, dtype=np.int64)

        self.pheromone_amount = pheromone_amount

    def move(self, action):
        for i in range(8):
            if action == i:
                self.y += ANT_MOVES[i][0]
                self.x += ANT_MOVES[i][1]

    def observe(self, grid, occupied_squares):
        return _observe(self, grid, occupied_squares)

    def clear_last(self):
        self.last_xys = np.array([(-1, -1)] * 10, dtype=np.int64)
