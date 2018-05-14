import serial
import time
import keyboard

port = "COM4"
baudrate=9600
CENTRE_x = int(60)
CENTRE_y = int(150)
SUCCESSFUL_EXCECUTION = 1
DEFAULT_MODE = 1

ser = serial.Serial(port, baudrate, timeout=0.1) # Serial comms arduino <--> Raspi
pan_pos = CENTRE_x
tilt_pos = CENTRE_y
list_modes = {1 : "<Demo Mode>", 2: "<Pump Mode>", 3: "<Vision Mode>"}

# Read serial buffer until empty
def read_until_empty():
    c = ser.read()
    word = b''
    while (c != b''):
        word += c
        c = ser.read()
    return word

# Check if string is a number
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Handler 1: Control rover movements using arrow keys
def move_rover(cmd, param):
    if cmd == "forward":
        ser.write(b"Move:forward:10 cm;")
    elif cmd == "backward":
        ser.write(b"Move:backward:10 cm;")
    elif cmd == "turn" and param == "left":
        ser.write(b"Move:turn:left;")
    else:
        ser.write(b"Move:turn:right;")
    #time.sleep(0.2)

# Function tells arduino to pump specified amount of liquid
def pump_liquid(amount_ml):
    message = "Pump:fertilize:" + amount_ml + " ml;"
    ser.write(message.encode(encoding='UTF-8'))

# In demo mode rover will fertilizer both sides of the garden row
def fertilize_row():
    pass

# HotKey Handler 1: Move pump with right arrow
def move_pump_right():
    global pan_pos
    if pan_pos >= 0:
        pan_pos -= 2
    message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
    ser.write(message.encode(encoding='UTF-8'))

# HotKey Handler 2: Move pump with left arrow
def move_pump_left():
    global pan_pos
    if pan_pos <= 140 :
       pan_pos += 2
    message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
    ser.write(message.encode(encoding='UTF-8'))

# HotKey Handler 3: Move pump with down arrow
def move_pump_down():
    global tilt_pos
    if tilt_pos >= 110:
        tilt_pos -= 2
    message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
    ser.write(message.encode(encoding='UTF-8'))

# HotKey Handler 4: Move pump with up arrow
def move_pump_up():
    global tilt_pos
    if tilt_pos <= 175:
        tilt_pos += 2
    message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
    ser.write(message.encode(encoding='UTF-8'))

class UserCmdLine:

    def __init__(self):
        self.current_mode = list_modes.get(DEFAULT_MODE)
        self.init_move()#initialize default mode = demo

    # Prompt User for a command
    def get_message(self):
        prompt = self.current_mode + " Enter Command: "
        return input(prompt)

    # Select mode. Modes = {Demo, Pump, Vision}
    def set_mode(self, mode):
        try:
            self.current_mode = list_modes[int(mode)]
            keyboard.unhook_all_hotkeys()
            if int(mode) == 1:
                self.init_move()
            else:
                self.init_pump()
        except KeyError:
            print("Invalid command! select a mode with keys [1,2 or 3] ")

    # Get current mode setting
    def get_mode(self):
        return self.current_mode

    # Initialise hotkeys for controlling pump orientation
    def init_pump(self):
        keyboard.add_hotkey('up', move_pump_up)
        keyboard.add_hotkey('down', move_pump_down)
        keyboard.add_hotkey('left', move_pump_left)
        keyboard.add_hotkey('right', move_pump_right)

    # Initialise hotkeys for controlling Rover in Demo mode
    def init_move(self):
        keyboard.add_hotkey('up', move_rover, args=('forward', '1 cm'))
        keyboard.add_hotkey('down', move_rover, args=('backward', '1 cm'))
        keyboard.add_hotkey('left', move_rover, args=('turn', 'left'))
        keyboard.add_hotkey('right', move_rover, args=('turn', 'right'))

def process_user_input(terminal, cnn_input):

    new_line = terminal.get_message()
    #Check if command was a mode setting 1,2 or 3
    if is_number(new_line):
        terminal.set_mode(new_line)
        return 0
    msg = new_line.split(' ')
    #Check if Mode == Demo Mode and command is valid
    if terminal.get_mode() == list_modes.get(DEFAULT_MODE):
        if msg[0] == 'Fertilize' and msg[1] == 'Row':
            fertilize_row()
        else:
            print("Invalid command! Use cmd[Fertilize Row]")
    #Check if Mode == Pump Mode and command is valid
    elif terminal.get_mode() == list_modes.get(2):
        if is_number(msg[0]) and msg[1] == "ml":
            pump_liquid(msg[0])
        else:
            print("Invalid command! Must have formart e.g. [10 ml]")
    # Check if Mode ==  Vision Mode
    else:
        if new_line == "detect":
            pass
        else:
            print("Invalid command! Use [detect]")

if __name__ == "__main__":
    main()
