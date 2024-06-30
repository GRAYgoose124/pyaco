import logging

from .window import AntColonyWindow


def main():
    DEBUG = True
    disabled_loggers = ["arcade", "numba", "matplotlib", "pyglet"]

    # Set up logging
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
    for name in disabled_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Debug vs. release settings
    if DEBUG:
        settings = debug_settings
    else:
        settings = default_settings

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
