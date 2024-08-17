import pyautogui
import time

# Function to perform the task
def perform_task(line):
    x=5
    pyautogui.click(x=40, y=183+x)  # new
    time.sleep(1)
    pyautogui.click(x=231, y=1001+x)  # location field
    time.sleep(1)
    pyautogui.click(x=204, y=800+x) # loc
    time.sleep(1)
    pyautogui.click(x=425, y=1000+x) # head
    time.sleep(1)
    pyautogui.click(x=452, y=750+x) # head
    time.sleep(1)
    pyautogui.click(x=877, y=1010+x) # sn field
    time.sleep(1)

    # Type the line from the file
    pyautogui.typewrite(line, interval=0.1)  # Adjust interval for typing speed
    time.sleep(2)
    pyautogui.click(x=898, y=933+x) # create
    time.sleep(2)
    pyautogui.click(x=1828, y=993+x) # apply

    time.sleep(2)  # Wait for 2 seconds

# Read from the file and perform the task for each line
def process_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any extra whitespace/newlines
            if line:  # Ensure the line is not empty
                time.sleep(1)  # Wait before processing the next line
                perform_task(line)

# Specify the path to your file here
filename = 'C:\\Users\\CanyonClark\\Downloads\\input.txt'
process_file(filename)
