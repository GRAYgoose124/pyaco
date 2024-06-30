import logging

from .window import AntColonyWindow


def main():
    # TODO: Fix non-square grids
    logging.basicConfig(level=logging.INFO)

    app = AntColonyWindow(
        window_size=(800, 800),
        resizable=False,
    )
    app._update_env_config(
        ant_count=500,
        its_per_frame=1,
        decay=0.99,
        pheremone=0.04,
        food_mul=1,
        n_food=10,
        square_size=5,
    )
    app.run()


if __name__ == "__main__":
    main()
