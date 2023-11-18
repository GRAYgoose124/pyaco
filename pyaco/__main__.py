import time
import arcade
import numpy as np

from .colony import AntColonyEnv


class AntColonyWindow(arcade.Window):
    def __init__(self, grid_size, ants):
        super().__init__(grid_size * 20, grid_size * 20, "Ant Colony Optimization")
        self.ITERATIONS_PER_FRAME = 10
        self.env = AntColonyEnv(grid_size, ants)
        self.env.reset()

        self.ant_sprites = arcade.SpriteList()
        self.grid_sprites = arcade.SpriteList()

        # Create grid sprites
        for i in range(grid_size):
            for j in range(grid_size):
                sprite = arcade.SpriteSolidColor(20, 20, arcade.color.WHITE)
                sprite.center_x = j * 20 + 20
                sprite.center_y = i * 20 + 20
                sprite.alpha = 255  # Set initial alpha, adjust as needed
                self.grid_sprites.append(sprite)

        # Create ant sprites
        for ant in self.env.ants:
            ant_sprite = arcade.SpriteSolidColor(10, 10, ant.color)
            ant_sprite.center_x = ant.x * 20 + 20
            ant_sprite.center_y = ant.y * 20 + 20
            self.ant_sprites.append(ant_sprite)

    def on_draw(self):
        arcade.start_render()
        self.grid_sprites.draw()
        self.ant_sprites.draw()

    def _update_transparency(self):
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                sprite_index = i * self.env.grid_size + j
                sprite = self.grid_sprites[sprite_index]
                pheromone_level = self.env.grid[i, j]
                try:
                    sprite.alpha = int(pheromone_level * 255)
                except ValueError:
                    print(pheromone_level)

    def on_update(self, delta_time):
        for _ in range(self.ITERATIONS_PER_FRAME):
            self.env.step()

        # Update ant sprites
        for ant, ant_sprite in zip(self.env.ants, self.ant_sprites):
            ant_sprite.center_x = ant.x * 20 + 20
            ant_sprite.center_y = ant.y * 20 + 20

        self._update_transparency()

        time.sleep(max(0, 1 / 60 - delta_time))


def main():
    app = AntColonyWindow(100, 100)
    app.run()


if __name__ == "__main__":
    main()
