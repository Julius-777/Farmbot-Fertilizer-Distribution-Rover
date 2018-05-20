 #!/usr/bin/python
import RaspiUI
import CNN_eval as ImageDetection
import time, os
import curses
from curses import wrapper

BACKSPACE = 8
ENTER = 10
newline = ''
c_prev = ord('1')


# Register events to occur depending on what User has input
def event_keys(stdscr, c, modes,):
    global c_prev, newline, current_mode, move_direction
    isNewLine = False
    move_direction = 'stop'
    if c == curses.KEY_UP:
        move_direction = 'up'

    elif c == curses.KEY_DOWN:
        move_direction = "down"

    elif c == curses.KEY_LEFT:
        move_direction = "left"

    elif c == curses.KEY_RIGHT:
        move_direction = "right"

    elif  (c_prev == ord('1') or c_prev == ord('2')\
            or c_prev == ord('3')) and c == ENTER:
        current_mode = modes[c_prev] # Select rover mode
        y,x = stdscr.getyx() # get current pos
        stdscr.move(y, 0)  # Move to start of line
        stdscr.clrtoeol()  # Erase the line to right of cursor
        stdscr.addstr(current_mode)
        stdscr.addstr(" Enter command: ")

    elif c == BACKSPACE:
        stdscr.addstr(chr(c))
        stdscr.delch()

    elif c == ENTER:
        stdscr.addstr(chr(c) + current_mode + " Enter command: ")
        isNewLine = True

    else: # input is a command (only valid if alphanumeric)
        success = False
        if 32<= c <=126:
            stdscr.addstr(chr(c)) # echo input
            newline +=  chr(c)

    c_prev = c
    return isNewLine

# Activte User interface for controlling System
def activate_UI(stdscr):
    # Set default UI parameters
    current_mode = modes[ord('1')]
    c = ENTER
    modes = {ord('1'): "<Demo Mode>",
             ord('2'): "<Pump Mode>",
             ord('3'): "<Vision Mode>"}
    # Initialize User Inferface Window
    stdscr.clear()
    stdscr.refresh()
    # Initialize image detection
    cnn = ImageDetection.PlantDetection()
    system = RaspiUI.SystemControl()
    # Escape programming using q
    while c is not ord('q'):
        # New command message has been input
        if event_keys(stdscr, c, modes) is True:
            RaspiUI.process_user_input(cnn, newline, current_mode)
            newline = ''

        elif move_direction is not 'stop':
            # Check for move commands
            RaspiUI.process_movements(move_direction, system, current_mode)
            time.sleep(0.2)

        stdscr.refresh() # Refresh the screen
        c = stdscr.getch() # returns key press values

def main():
    wrapper(activate_UI) # Initialize and restores terminal before/after use

if __name__ == "__main__":
    main()
