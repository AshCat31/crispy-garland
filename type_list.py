import pyautogui
import time

# Function to read strings from files
def read_strings_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

# Main function to simulate typing and pressing Enter
def simulate_typing(strings):
    for string in strings:
        # pyautogui.click()  # Ensure the cursor is in the active field
        pyautogui.typewrite(string)  # Type the string
        time.sleep(0.5)  # Optional: Adjust the delay between typing and pressing Enter
        pyautogui.press('enter')  # Simulate pressing Enter
        time.sleep(1)  # Optional: Adjust the delay between Enter presses

if __name__ == "__main__":
    filename = 'C:\\Users\\CanyonClark\\Downloads\\input.txt'  # Replace with your file name
    strings = read_strings_from_file(filename)
    
    time.sleep(3)

    try:
        simulate_typing(strings)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
