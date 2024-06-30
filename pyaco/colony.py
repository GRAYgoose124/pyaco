import gym
from gym import spaces
import numpy as np
import arcade
import numba
import logging
from numba.typed import List

from .ant import Ant

log = logging.getLogger(__name__)


def random_color():
    return (
        np.random.randint(0, 255),
        np.random.randint(64, 255),
        np.random.randint(0, 255),
    )


@numba.jit(nopython=True)
def _step(
    grid, grid_size, ants, pheromone_decay_rate, foods, remains
) -> tuple[np.ndarray, float, bool, dict]:
    reward = 0
    ate_food = None
    occupied_squares = np.zeros(grid.shape, dtype=np.int64)
    for ant in ants:
        occupied_squares[ant.y, ant.x] = 1
    for food in foods:
        occupied_squares[food[1], food[0]] = 2

    for ant in ants:
        # Get the local grid around the ant
        action = ant.observe(grid, occupied_squares)
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

        for i, food in enumerate(foods):
            food_x, food_y = food
            if ant.x == food_x and ant.y == food_y:
                ant.x = np.random.randint(0, grid_size[0])
                ant.y = np.random.randint(0, grid_size[1])
                ant.last_x, ant.last_y = -1, -1
                ant.second_last_x, ant.second_last_y = -1, -1
                remains[i] -= 1
                if remains[i] == 0:
                    ate_food = food

    # Pheromone evaporation
    grid *= pheromone_decay_rate

    done = False

    result = grid, reward, done, ate_food
    return result


class AntColonyEnv(gym.Env):
    def __init__(
        self,
        grid_size,
        num_ants,
        decay_rate=0.999,
        ant_pheromone_amount=0.01,
        food_mul=10,
        n_food=10,
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
        self.ant_count = num_ants

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.pheromone_decay_rate = decay_rate

        # dtype needs to be tuple
        self.foods = List()
        self.remains = List()
        for _ in range(int(abs(n_food))):
            self.foods.append(
                np.array(
                    (
                        np.random.randint(0, grid_size[0]),
                        np.random.randint(0, grid_size[1]),
                    ),
                )
            )
            self.remains.append(self.ant_count * food_mul)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=grid_size, dtype=np.float32
        )

    def step(self):
        self.grid, reward, done, ate_food = _step(
            self.grid,
            self.grid_size,
            self.ants,
            self.pheromone_decay_rate,
            self.foods,
            self.remains,
        )

        if ate_food is not None:
            log.debug(f"Food eaten at {ate_food}, remaining {self.remains}")
            # Find the index of the item to remove
            index_to_remove = None
            for i, food in enumerate(self.foods):
                if np.array_equal(food, ate_food):
                    log.info(f"Found food to remove at {food}")
                    index_to_remove = i
                    break

            if index_to_remove is not None:
                self.foods.pop(index_to_remove)
                # add new food
                self.foods.append(
                    np.array(
                        (
                            np.random.randint(0, self.grid_size[0]),
                            np.random.randint(0, self.grid_size[1]),
                        ),
                    )
                )

        return self.grid, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.grid = np.zeros(self.grid_size)
        return self.grid

    def close(self):
        if hasattr(self, "window"):
            arcade.close_window()
