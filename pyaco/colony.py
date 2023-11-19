import gym
from gym import spaces
import numpy as np
import arcade
import numba
from numba.typed import List

from .ant import Ant


def random_color():
    return (
        np.random.randint(0, 255),
        np.random.randint(64, 255),
        np.random.randint(0, 255),
    )


@numba.jit(nopython=True)
def weighted_random_choice(choices, probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    random_choice = np.random.rand()
    for i, prob in enumerate(cumulative_probabilities):
        if random_choice < prob:
            return choices[i]
    return choices[-1]


@numba.jit(nopython=True)
def observe(ant, local_grid: np.ndarray, occupied_squares: np.ndarray):
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    pheromone_values = np.empty(len(moves), dtype=np.float64)

    center_y, center_x = 1, 1  # Center of the 3x3 grid

    # Calculate the previous move direction
    prev_dy, prev_dx = 0, 0
    if ant.last_x != -1 and ant.last_y != -1:
        prev_dy = ant.y - ant.last_y
        prev_dx = ant.x - ant.last_x

    for idx, (dy, dx) in enumerate(moves):
        grid_y, grid_x = center_y + dy, center_x + dx

        if 0 <= grid_y < 3 and 0 <= grid_x < 3:
            pheromone_value = local_grid[grid_y, grid_x]

            # Bias towards forward motion
            if dy == prev_dy and dx == prev_dx:
                pheromone_value *= 1.5  # Example: Increase pheromone value by 20%

            pheromone_values[idx] = pheromone_value
        else:
            pheromone_values[idx] = -1.0  # Invalid moves get a negative value

    probabilities = np.exp(pheromone_values)
    probabilities /= np.sum(probabilities)
    action = weighted_random_choice(np.arange(8), probabilities)
    return action


@numba.jit(nopython=True)
def _step(
    grid, grid_size, ants, pheromone_decay_rate, food_x, food_y
) -> (np.ndarray, float, bool, dict):
    reward = 0
    new_food = False
    occupied_squares = np.zeros(grid.shape, dtype=np.bool_)
    for ant in ants:
        occupied_squares[ant.y, ant.x] = True

    for ant in ants:
        # Get the local grid around the ant
        local_grid = grid[max(0, ant.y - 2) : ant.y + 3, max(0, ant.x - 2) : ant.x + 3]
        action = observe(ant, local_grid, occupied_squares)
        ant.move(action)
        # Wrap around the grid
        ant.x = ant.x % grid_size[0]
        ant.y = ant.y % grid_size[1]
        # Store the last position
        ant.last_x, ant.last_y = ant.x, ant.y
        ant.second_last_x, ant.second_last_y = ant.last_x, ant.last_y
        # Deposit pheromone
        grid[ant.last_y, ant.last_x] = min(
            1.0, grid[ant.last_y, ant.last_x] + ant.pheromone_amount
        )
        if ant.x == food_x and ant.y == food_y:
            reward += 1
            ant.x = np.random.randint(0, grid_size[0])
            ant.y = np.random.randint(0, grid_size[1])
            ant.last_x, ant.last_y = -1, -1
            ant.second_last_x, ant.second_last_y = -1, -1
            new_food = True

    # Pheromone evaporation
    grid *= pheromone_decay_rate

    done = False

    result = grid, reward, done, new_food
    return result


class AntColonyEnv(gym.Env):
    def __init__(
        self, grid_size, num_ants, decay_rate=0.999, ant_pheromone_amount=0.01
    ):
        super(AntColonyEnv, self).__init__()

        self.ants = List()
        for _ in range(num_ants):
            self.ants.append(
                Ant(
                    np.random.randint(0, grid_size[0]),
                    np.random.randint(0, grid_size[1]),
                    color=random_color(),
                    pheromone_amount=ant_pheromone_amount,
                )
            )

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.pheromone_decay_rate = decay_rate

        self.food_x = np.random.randint(0, grid_size[0])
        self.food_y = np.random.randint(0, grid_size[1])

        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=grid_size, dtype=np.float32
        )

    def step(self):
        self.grid, reward, done, new_food = _step(
            self.grid,
            self.grid_size,
            self.ants,
            self.pheromone_decay_rate,
            self.food_x,
            self.food_y,
        )

        if new_food:
            self.food_x = np.random.randint(0, self.grid_size[0])
            self.food_y = np.random.randint(0, self.grid_size[1])

        return self.grid, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.grid = np.zeros(self.grid_size)
        return self.grid

    def close(self):
        if hasattr(self, "window"):
            arcade.close_window()
