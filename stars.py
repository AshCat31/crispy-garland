import curses
import random
import time

def initialize_stars(grid_width, grid_height, num_stars):
    grid_size = grid_width * grid_height
    stars = random.sample(range(grid_size), num_stars)
    return stars

def main(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()

    grid_width = 233  # Adjust the width of the grid
    grid_height = 40  # Adjust the height of the grid
    num_stars = int(.01 * grid_height * grid_width)
    stars_indices = initialize_stars(grid_width, grid_height, num_stars)
    star_chars = {}  # Dictionary to store character for each star

    for star_index in stars_indices:
        star_chars[star_index] = random.choice(['*', 'X', '+', '.', '@'])

    while True:
        stdscr.clear()
        for i in range(grid_height):
            for j in range(grid_width):
                index = i * grid_width + j
                if index in stars_indices:
                    # Check if the star should change to a new character
                    if random.random() < 0.5:
                        star_chars[index] = random.choice(['*', 'X', '+', '.', '@'])
                    stdscr.addch(i, j, star_chars[index])
                else:
                    stdscr.addch(i, j, ' ')
        stdscr.refresh()
        time.sleep(0.3)  # Adjust the delay between frames here

if __name__ == "__main__":
    curses.wrapper(main)
