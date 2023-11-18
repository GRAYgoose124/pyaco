import time
import arcade

from .colony import AntColonyEnv


class AntColonyWindow(arcade.Window):
    def __init__(self, grid_size, ants):
        super().__init__(grid_size * 20, grid_size * 20, "Ant Colony Optimization")
        self.ITERATIONS_PER_FRAME = 10
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
        for _ in range(self.ITERATIONS_PER_FRAME):
            self.env.step()
        time.sleep(max(0, 1 / 60 - delta_time))


def main():
    app = AntColonyWindow(50, 100)
    app.run()


if __name__ == "__main__":
    main()
