import pyautogui

# Display mouse xcursor position continuously
print("Press Ctrl-C to quit.")
try:
    while True:
        # Get and print the mouse coordinatess
        x, y = pyautogui.position()
        position_str = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(position_str, end='\r')
except KeyboardInterrupt:
    print('\nDone.')
