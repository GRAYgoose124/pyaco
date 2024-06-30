import time
import arcade
import numpy as np
import logging

from .colony import AntColonyEnv

log = logging.getLogger(__name__)


class AntColonyWindow(arcade.Window):
    def __init__(
        self,
        window_size=(800, 800),
        resizable=False,
    ):
        super().__init__(
            window_size[0],
            window_size[1],
            "Ant Colony Optimization",
            resizable=resizable,
        )

        self._update_env_config()
        self._calculate_grid_size(window_size)
        self._initialize_env(new=True)
        self._create_sprites()

    @staticmethod
    def _default_env_config():
        return {
            "square_size": 5,
            "its_per_frame": 10,
            "decay": 0.995,
            "pheremone": 0.04,
            "ant_count": 50,
            "food_mul": 10,
            "n_food": 10,
        }

    def _update_env_config(self, **kwargs):
        if not hasattr(self, "env_config"):
            self.env_config = self._default_env_config()

        for key, value in kwargs.items():
            if key in self.env_config:
                self.env_config[key] = value
            else:
                raise ValueError(f"Invalid key {key}")

    def _calculate_grid_size(self, window_size=None):
        square_size = self.env_config["square_size"]
        if window_size is None:
            window_size = self.get_size()
        squares_width = window_size[0] // square_size
        squares_height = window_size[1] // square_size

        self.square_size = square_size
        self.grid_size = (squares_width, squares_height)

    def _initialize_env(self, new=False):
        if new:
            self.env = AntColonyEnv(
                self.grid_size,
                self.env_config["ant_count"],
                decay_rate=self.env_config["decay"],
                ant_pheromone_amount=self.env_config["pheremone"],
                food_mul=self.env_config["food_mul"],
                n_food=self.env_config["n_food"],
            )
        self.env.reset()

    def _create_sprites(self):
        self.ant_sprites = arcade.SpriteList()
        self.grid_sprites = arcade.SpriteList()
        self.food_sprites = arcade.SpriteList()

        # Create grid sprites
        for i in range(self.grid_size[0]):  # Note the change here for y-coordinate
            for j in range(self.grid_size[1]):  # Note the change here for x-coordinate
                sprite = arcade.SpriteSolidColor(
                    self.square_size, self.square_size, arcade.color.WHITE
                )
                sprite.center_x = i * self.square_size + self.square_size / 2
                sprite.center_y = j * self.square_size + self.square_size / 2
                self.grid_sprites.append(sprite)

        # Create ant sprites
        for ant in self.env.ants:
            ant_sprite = arcade.SpriteSolidColor(
                self.square_size // 2, self.square_size // 2, ant.color
            )
            ant_sprite.alpha = 200
            ant_sprite.center_x = ant.x * self.square_size + self.square_size / 2
            ant_sprite.center_y = ant.y * self.square_size + self.square_size / 2
            self.ant_sprites.append(ant_sprite)

        # Create food sprites
        for food in self.env.foods:
            food_sprite = arcade.SpriteSolidColor(
                self.square_size * 2, self.square_size * 2, arcade.color.GREEN
            )
            food_sprite.center_x = food[0] * self.square_size + self.square_size / 2
            food_sprite.center_y = food[1] * self.square_size + self.square_size / 2
            self.food_sprites.append(food_sprite)

    def on_draw(self):
        arcade.start_render()
        self.grid_sprites.draw()
        self.ant_sprites.draw()
        self.food_sprites.draw()

    def _update_transparency(self):
        for i in range(self.env.grid_size[0]):  # Rows
            for j in range(self.env.grid_size[1]):  # Columns
                sprite_index = i + j * self.env.grid_size[0]

                sprite = self.grid_sprites[sprite_index]
                pheromone_level = self.env.grid[i, j]
                pheromone_level = max(0, min(1, pheromone_level))
                sprite.alpha = int(pheromone_level * 255)
        # faster:
        # for i, sprite in enumerate(self.grid_sprites):
        #     pheromone_level = self.env.grid[
        #         i // self.grid_size[0], i % self.grid_size[0]
        #     ]
        #     pheromone_level = max(0, min(1, pheromone_level))
        #     sprite.alpha = int(pheromone_level * 255)

    def on_update(self, delta_time):
        for _ in range(self.env_config["its_per_frame"]):
            self.env.step()

        # Update ant sprites
        for ant, ant_sprite in zip(self.env.ants, self.ant_sprites):
            ant_sprite.center_x = ant.x * self.square_size + self.square_size / 2
            ant_sprite.center_y = ant.y * self.square_size + self.square_size / 2

        # Update food sprites
        for food, food_sprite in zip(self.env.foods, self.food_sprites):
            food_sprite.center_x = food[0] * self.square_size + self.square_size / 2
            food_sprite.center_y = food[1] * self.square_size + self.square_size / 2

        self._update_transparency()

        # TODO: more efficient...
        # time.sleep(max(0, 1 / 60 - delta_time))

    def on_resize(self, width: float, height: float):
        self._calculate_grid_size((width, height))
        self._initialize_env()
        self._create_sprites()
