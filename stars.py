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

    grid_width = 150  # Adjust the width of the grid
    grid_height = 40  # Adjust the height of the grid
    num_stars = int(.01* grid_height*grid_width)
    stars_indices = initialize_stars(grid_width, grid_height, num_stars)

    while True:
        stdscr.clear()
        for i in range(grid_height):
            for j in range(grid_width):
                if i * grid_width + j in stars_indices:
                    stdscr.addch(i, j, random.choice(['*', 'X', '+', '.', '@']))
                else:
                    stdscr.addch(i, j, ' ')
        stdscr.refresh()
        time.sleep(0.5)  # Adjust the delay between frames here

if __name__ == "__main__":
    curses.wrapper(main)
