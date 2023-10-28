import time
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import arcade
import numba
from numba import int32, int64, float64, boolean


def random_color():
    return (
        np.random.randint(0, 255),
        np.random.randint(64, 255),
        np.random.randint(0, 255),
    )


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

        self.pheromone_amount = 0.05

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
    # Define possible moves: (dy, dx)
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    pheromone_values = np.empty(0, dtype=np.float64)

    for idx, (dy, dx) in enumerate(moves):
        # Check if the move is within the local grid boundaries
        if (
            0 <= ant.y + dy < local_grid.shape[0]
            and 0 <= ant.x + dx < local_grid.shape[1]
            and not occupied_squares[ant.y + dy, ant.x + dx]
            and (ant.last_y == -1 or ant.y + dy != ant.last_y)
            and (ant.last_x == -1 or ant.x + dx != ant.last_x)
            and (
                ant.last_y == -1
                or ant.y + dy != ant.last_y
                or ant.y + dy != ant.second_last_y
            )
            and (
                ant.last_x == -1
                or ant.x + dx != ant.last_x
                or ant.x + dx != ant.second_last_x
            )
        ):
            pheromone_values = np.append(
                pheromone_values, local_grid[ant.y + dy, ant.x + dx]
            )
        else:
            pheromone_values = np.append(
                pheromone_values, -1.0
            )  # Invalid moves get a negative value

    # Find the indices of the maximum pheromone values
    max_indices = np.where(pheromone_values == pheromone_values.max())[0]

    # Choose a move based on the maximum pheromone values
    action = np.random.choice(max_indices)
    return action


class AntColonyEnv(gym.Env):
    def __init__(self, grid_size, num_ants):
        super(AntColonyEnv, self).__init__()

        self.ants = [
            Ant(
                np.random.randint(0, grid_size),
                np.random.randint(0, grid_size),
                color=random_color(),
            )
            for _ in range(num_ants)
        ]

        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.pheromone_decay_rate = 0.99

        self.food_x = np.random.randint(0, grid_size)
        self.food_y = np.random.randint(0, grid_size)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )

    @numba.jit(forceobj=True)
    def step(self):
        reward = 0
        occupied_squares = np.zeros_like(self.grid, dtype=bool)
        for ant in self.ants:
            occupied_squares[ant.y, ant.x] = True

        for ant in self.ants:
            # Get the local grid around the ant
            local_grid = self.grid[
                max(0, ant.y - 1) : ant.y + 2, max(0, ant.x - 1) : ant.x + 2
            ]
            action = observe(ant, local_grid, occupied_squares)
            ant.move(action)
            # Wrap around the grid
            ant.x = ant.x % self.grid_size
            ant.y = ant.y % self.grid_size
            # Store the last position
            ant.last_x, ant.last_y = ant.x, ant.y
            ant.second_last_x, ant.second_last_y = ant.last_x, ant.last_y
            # Deposit pheromone
            self.grid[ant.last_y, ant.last_x] += ant.pheromone_amount
            if ant.x == self.food_x and ant.y == self.food_y:
                reward += 1
                ant.x = np.random.randint(0, self.grid_size)
                ant.y = np.random.randint(0, self.grid_size)
                ant.last_x, ant.last_y = -1, -1
                ant.second_last_x, ant.second_last_y = -1, -1

        # Pheromone evaporation
        self.grid *= self.pheromone_decay_rate

        done = False

        return self.grid, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.grid = np.zeros((self.grid_size, self.grid_size))
        return self.grid

    def close(self):
        if hasattr(self, "window"):
            arcade.close_window()


class AntColonyWindow(arcade.Window):
    def __init__(self, grid_size, ants):
        super().__init__(grid_size * 20, grid_size * 20, "Ant Colony Optimization")
        self.env = AntColonyEnv(grid_size, ants)
        self.env.reset()

    def on_draw(self):
        arcade.start_render()
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                color = (255, 255, 255, int(self.env.grid[i, j] * 255))
                arcade.draw_rectangle_filled(j * 20 + 20, i * 20 + 20, 20, 20, color)

        # Draw ants
        for ant in self.env.ants:
            arcade.draw_circle_filled(ant.x * 20 + 20, ant.y * 20 + 20, 5, ant.color)

        arcade.draw_circle_filled(
            self.env.food_x * 20 + 20, self.env.food_y * 20 + 20, 10, (0, 255, 0)
        )

    def on_update(self, delta_time):
        self.env.step()
        # sleep for 60fps, accounting for time spent on rendering
        # time.sleep(max(0, 1 / 60 - delta_time))


def main():
    app = AntColonyWindow(50, 100)
    app.run()


if __name__ == "__main__":
    main()
