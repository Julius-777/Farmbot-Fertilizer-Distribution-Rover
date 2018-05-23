 #!/usr/bin/python
import RaspiUI
import CNN_eval as ImageDetection
import time, os
import curses
from curses import wrapper

BACKSPACE = 263
ENTER = 10
c_prev = ord('1')


# Register events to occur depending on what User has input
def event_keys(stdscr, c, modes,):
    global c_prev, newline, current_mode
    global boundary,  move_direction
    isNewLine = False
    move_direction = 'stop'
    y,x = stdscr.getyx() # get current cursor pos
    
    if c == curses.KEY_UP:
        move_direction = 'up'

    elif c == curses.KEY_DOWN:
        move_direction = "down"

    elif c == curses.KEY_LEFT:
        move_direction = "left"

    elif c == curses.KEY_RIGHT:
        move_direction = "right"

    # Change the mode setting if 1, 2 or 3 is pressed
    elif (c_prev == ord('1') or c_prev == ord('2') or c_prev == ord('3'))\
            and c == ENTER and x == (boundary + 1):
            current_mode = modes[c_prev] # Select rover mode
            stdscr.move(y, 0)  # Move to start of line
            stdscr.clrtoeol()  # Erase the line to right of cursor
            stdscr.addstr(current_mode)
            stdscr.addstr(" Enter command: ")
	    boundary = len(current_mode + " Enter command: ")
	    newline = ''

    # If backspace pressed then delete
    elif c == BACKSPACE:
	# Delte characters up to prompt command
	if stdscr.getyx()[1] > boundary:
        	stdscr.addstr(chr(8))
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
def activate_UI(stdscr, cnn, system):
    # Set default UI parameters
    global newline, current_mode, boundary
    c = ENTER
    modes = {ord('1'): "<Demo Mode>",
             ord('2'): "<Pump Mode>",
             ord('3'): "<Vision Mode>"}
    current_mode = modes[ord('1')]
    newline = ''
    prev_time = time.time() # last time an arrow key was pressed
    boundary = len(current_mode + " Enter command: ")

    # Initialize User Inferface Window    
    stdscr.clear()
    stdscr.refresh()

    # Escape programming using q
    while c is not ord('q'):
        # New command message has been input
        if event_keys(stdscr, c, modes) is True:
            stdscr.addstr(newline)
            response = RaspiUI.process_user_input(cnn, newline, system, current_mode, stdscr)
            newline = ''
            # One  message
            if type(response) is str:
                message = response + "\n" + current_mode + " Enter command: "
                stdscr.addstr(message)
            # Two messages
            elif type(response) is list:
                message = response[0] + "\n"+ response[1] + "\n"+ current_mode + " Enter command: "
                stdscr.addstr(message)
            # Incorrect input
            else:
                stdscr.addstr("Error!!\n") 
	# Check if arrow key was pressed
        elif move_direction != 'stop':
            # Send move command every 0.2 seconds 
	    if time.time() - prev_time > 0.02:
            	RaspiUI.process_movements(move_direction, system, current_mode, stdscr)
		prev_time = time.time()

        stdscr.refresh() # Refresh the screen
        c = stdscr.getch() # returns key press values

def main():
    # Initialize image detection and System control class
    cnn = ImageDetection.PlantDetection()
    system = RaspiUI.SystemControl()

    # Initialize and restores terminal before/after use
    wrapper(activate_UI, cnn, system) 

if __name__ == "__main__":
    main()
