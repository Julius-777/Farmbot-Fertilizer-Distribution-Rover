#!/usr/bin/python
import serial
import time

port = {1:"/dev/ttyUSB0", 2:"/dev/ttyUSB1"}
baudrate=9600
CENTRE_x = int(60)
CENTRE_y = int(150)
SUCCESSFUL_EXCECUTION = 1
DEFAULT_MODE = 1
DISPLAY_ON = True # display the stages in plant recognition on terminal
global s
# Serial comms arduino <--> Raspi
try:
    ser = serial.Serial(port[1], baudrate, timeout=0.1)
except SerialException:
    ser = serial.Serial(port[2], baudrate, timeout=0.1)

pan_pos = CENTRE_x
tilt_pos = CENTRE_y
list_modes = {1 : "<Demo Mode>", 2: "<Pump Mode>", 3: "<Vision Mode>"}
LOG = {"Plant": None, "Stage": None, "used_fertilizer": None,
                            "Tank_Level": 1500} # format of logging message

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

# Function tells arduino to pump specified amount of liquid
def pump_liquid(amount_ml):
    message = "Pump:fertilize:" + amount_ml + " ml;"
    ser.write(message.encode(encoding='UTF-8'))

# In demo mode rover will fertilizer both sides of the garden row
def fertilize_row():
    pass

# Control Movements of fertilizer system on rover
class SystemControl:
    # Control rover movements using arrow keys
    def move_rover(self, cmd, param):
        if cmd == "forward":
            ser.write(b"Move:forward:3 cm;")

        elif cmd == "backward":
            ser.write(b"Move:backward:3 cm;")

        elif cmd == "turn" and param == "left":
            ser.write(b"Move:turn:left;")

        else:
            ser.write(b"Move:turn:right;")

    # Move pump with right arrow
    def move_pump_right(self):
        global pan_pos, s
        if pan_pos >= 0:
            pan_pos -= 2
        message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
        ser.write(message.encode(encoding='UTF-8'))

    # Move pump with left arrow
    def move_pump_left(self):
        global pan_pos, s
        if pan_pos <= 140 :
           pan_pos += 2
        message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
        ser.write(message.encode(encoding='UTF-8'))

    # Move pump with down arrow
    def move_pump_down(self):
        global tilt_pos
        if tilt_pos >= 110:
            tilt_pos -= 2
        message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
        ser.write(message.encode(encoding='UTF-8'))

    # Move pump with up arrow
    def move_pump_up(self):
        global tilt_pos
        if tilt_pos <= 175:
            tilt_pos += 2
        message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
        ser.write(message.encode(encoding='UTF-8'))

# Process arrow key presses from terminal to move peripherals
def process_movements(direction, system, current_mode, st):
    # Demo Mode
    global s
    s = st
    if current_mode == list_modes[1]:
        if direction == 'up': # Up arrow = forward
            system.move_rover('forward', None)

        elif direction == 'down': # Down arrow = backward
            system.move_rover('backward', None)

        else:
            # Turn in specified direction
            system.move_rover('turn', direction)

    # Pump or Vision Mode
    elif current_mode == list_modes[2] or  current_mode == list_modes[3]:
        if direction == 'left':
            system.move_pump_left()

        elif direction == 'right':
            system.move_pump_right()

        elif direction == 'up':
            system.move_pump_down() # Tilt is reversed

        elif direction == 'down':
            system.move_pump_up()

# Process typed user input commands from terminal
def process_user_input(cnn, new_line, current_mode):
    msg = new_line.split(' ')
    #Check if Mode == Demo Mode and command is valid
    if current_mode == list_modes.get(DEFAULT_MODE):
        if msg[0] == 'fertilize' and msg[1] == 'row':
            fertilize_row()
            return ''
        else:
            return "Invalid command! Use cmd[fertilize row]"
    #Check if Mode == Pump Mode and command is valid
    elif current_mode == list_modes.get(2):
        if is_number(msg[0]) and msg[1] == "ml":
            pump_liquid(msg[0])
            return ''
        else:
            return "Invalid command! Must have formart e.g. [10 ml]"
    # Check if Mode ==  Vision Mode
    else:
        if new_line == "detect":
            predictions = [0,0]
            # Accept detection with a score above 80% accuracy
            while predictions[1] < 0.90:
                image_array = cnn.prepare_images(from_camera=True) # Get piCam image as numpy array
                predictions = cnn.get_predictions(image_array,
                                                 data_type="from_camera",
                                                 display=True)# display all predictions
                time.sleep(0.3) # check 3 frames per second
            #return  "Predicted label - %s, Score: [%5f]" % (predictions[0],
            #                                              predictions[1])

        else:
            return "Invalid command! Use [detect]"
