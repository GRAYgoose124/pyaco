import time
import arcade
import numpy as np

from .colony import AntColonyEnv


class AntColonyWindow(arcade.Window):
    def __init__(self, window_size=(1280, 720), ant_count=50, square_size=5):
        self.square_size = square_size

        # Calculate how many squares can fit in width and height
        squares_width = window_size[0] // square_size
        squares_height = window_size[1] // square_size

        grid_size = (squares_width, squares_height)

        # Initialize the window with the original window size
        super().__init__(
            window_size[0],
            window_size[1],
            "Ant Colony Optimization",
        )
        self.grid_size = grid_size
        self.ITERATIONS_PER_FRAME = 10
        self.DECAY_RATE = 0.995
        self.ANT_PHEREMONE_AMOUNT = 0.04
        self.env = AntColonyEnv(
            grid_size,
            ant_count,
            decay_rate=self.DECAY_RATE,
            ant_pheromone_amount=self.ANT_PHEREMONE_AMOUNT,
        )
        self.env.reset()

        self.ant_sprites = arcade.SpriteList()
        self.grid_sprites = arcade.SpriteList()

        # Create grid sprites
        for i in range(grid_size[0]):  # Note the change here for y-coordinate
            for j in range(grid_size[1]):  # Note the change here for x-coordinate
                sprite = arcade.SpriteSolidColor(
                    square_size, square_size, arcade.color.WHITE
                )
                sprite.center_x = i * square_size + square_size / 2
                sprite.center_y = j * square_size + square_size / 2
                self.grid_sprites.append(sprite)

        # Create ant sprites
        for ant in self.env.ants:
            ant_sprite = arcade.SpriteSolidColor(
                square_size // 2, square_size // 2, ant.color
            )
            ant_sprite.alpha = 200
            ant_sprite.center_x = ant.x * square_size + square_size / 2
            ant_sprite.center_y = ant.y * square_size + square_size / 2
            self.ant_sprites.append(ant_sprite)

    def on_draw(self):
        arcade.start_render()
        self.grid_sprites.draw()
        self.ant_sprites.draw()

    def _update_transparency(self):
        for i in range(self.env.grid_size[0]):  # Rows
            for j in range(self.env.grid_size[1]):  # Columns
                sprite_index = i * self.env.grid_size[1] + j

                sprite = self.grid_sprites[sprite_index]
                pheromone_level = self.env.grid[i, j]
                pheromone_level = max(0, min(1, pheromone_level))
                sprite.alpha = int(pheromone_level * 255)

    def on_update(self, delta_time):
        for _ in range(self.ITERATIONS_PER_FRAME):
            self.env.step()

        # Update ant sprites
        for ant, ant_sprite in zip(self.env.ants, self.ant_sprites):
            ant_sprite.center_x = ant.x * self.square_size + self.square_size / 2
            ant_sprite.center_y = ant.y * self.square_size + self.square_size / 2

        self._update_transparency()

        time.sleep(max(0, 1 / 60 - delta_time))


def main():
    # TODO: Fix non-square grids
    app = AntColonyWindow(window_size=(800, 800), ant_count=500)
    app.run()


if __name__ == "__main__":
    main()
