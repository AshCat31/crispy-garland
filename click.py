import pyautogui
import time

# Define the coordinates for clicks (adjust these according to your screen)
click_coordinates = [
    # (75, 235),
    # (650, 930),
    # (1885, 180),
# (1391,584)
# (613, 699),

# (58, 242),
# (669, 637), # web, no components
# (520, 950),
# (1070,660),
# (966,656),  # epoxy
(50,250),
# (524,745),
(1880, 180)  # next
]

x = 20  # Change this to the number of times you want to perform the clicks
print("Get to odoo!")
time.sleep(1)

# Loop through the specified number of times
for _ in range(x):
    # Perform 3 clicks with pauses in between
    for coord in click_coordinates:
        pyautogui.click(coord[0], coord[1])
        if coord[0] == 966:
            pyautogui.typewrite("0")
        time.sleep(3)  # Adjust the sleep time (in seconds) as needed

    # Extra pause between iterations
    time.sleep(.8)  # Adjust as needed
