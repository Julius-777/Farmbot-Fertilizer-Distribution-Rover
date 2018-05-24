#!/usr/bin/python -W ignore::DeprecationWarning 
import serial
import time
import json, requests
import cropNutrients 

grafana_path = 'https://farmbot-data.uqcloud.net/apix/record'  # URL logging path
ports = {1:"/dev/ttyUSB0", 2:"/dev/ttyUSB1"}
baudrate=9600
# servo position

CENTRE_x = int(60) #default  position pan
CENTRE_y = int(150) #default position tilt
pan_pos = CENTRE_x
tilt_pos = CENTRE_y

SUCCESSFUL_EXCECUTION = 1
DEFAULT_MODE = 1 # Demo Mode
DISPLAY_ON = True # display the stages in plant recognition on terminal
SCAN_ON = True # scan for plant features
# Mode settings for User commands
list_modes = {1 : "<Demo Mode>", 2: "<Pump Mode>", 3: "<Vision Mode>"}
# Liquid Tank Levels
TANK_LEVEL =  int(1500)

# Serial comms arduino <--> Raspi
try:
    ser = serial.Serial(ports[1], baudrate, timeout=0.1)
except Exception:
    ser = serial.Serial(ports[2], baudrate, timeout=0.1)

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
    global TANK_LEVEL 
    if TANK_LEVEL> 0:
        message = "Pump:fertilize:" + str(amount_ml) + " ml;"
        ser.write(message.encode(encoding='UTF-8'))
        TANK_LEVEL -= int(amount_ml)
        return "Tank has ["+ str(TANK_LEVEL)+" ml] left"
    else:
        return  "Tank Empty !! Refill Tank"

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
        global pan_pos
        if pan_pos >= 0:
            pan_pos -= 4
            message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
            ser.write(message.encode(encoding='UTF-8'))
            return True 
        return False # Max angle reached
    
    # Move pump with left arrow
    def move_pump_left(self):
        global pan_pos 
        if pan_pos <= 140 :
            pan_pos += 4
            message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
            ser.write(message.encode(encoding='UTF-8'))
            return True
        return False

    # Move pump with down arrow
    def move_pump_down(self):
        global tilt_pos
        if tilt_pos >= 110:
            tilt_pos -= 4
            message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
            ser.write(message.encode(encoding='UTF-8'))
            return True
        return False

    # Move pump with up arrow
    def move_pump_up(self):
        global tilt_pos
        if tilt_pos <= 175:
            tilt_pos += 4
            message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
            ser.write(message.encode(encoding='UTF-8'))
            return True
        return False
    
    # Recentre pump 
    def move_pump_centre(self, direction):
        global tilt_pos, pan_pos
        global CENTRE_x, CENTRE_y
        if direction == 'pan':
            pan_pos = CENTRE_x
        elif direction == 'tilt':
            tilt_pos = CENTRE_y
        else:
            pan_pos = CENTRE_x
            tilt_pos = CENTRE_y
      
        message = "PanTilt:angle:" + str(pan_pos) + ',' + str(tilt_pos) + ';'
        ser.write(message.encode(encoding='UTF-8'))

# Log data to Farmdata Grafana Database for visualization using json string 
def log_data(**args):
    plant = args.get("plant", None)
    ml = args.get("liquid", None)
    current_stage = args.get("stage", None)

    ts = int(time.time())
    myUSER = "s4358870"
    stages = {1:"germination", 2:"growth", 3:"flowering"} # Plant growth stages
    myPASS = "RelativelyDeathLanesAptly"

    req = json.dumps({"userid":myUSER, "passhash":myPASS,
      "timestamp": int(ts *1e3), # time
      "tags":{"plant":plant, "stage": current_stage}, # meta data
      "data":{"liquid_ml":ml} # data
    })
    
    # Send data to farmbot
    resp = requests.post(grafana_path, req) # Send JSON string
    return resp

# Process arrow key presses from terminal to move peripherals
def process_movements(direction, system, current_mode):
    # Demo Mode
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

# detect plant type (If scan=True system looks up and down)
def detect_crop(cnn, scan_on):
    global system, stream
    prediction = (0,0)
    scanup = True
    scandown = True
    # Accept detection with a score above 80% accuracy
    while prediction[1] < 0.70:
        image_array = cnn.prepare_images(from_camera=True) # Get piCam image as numpy array
        prediction = cnn.get_predictions(image_array,
                                          data_type="from_camera",
                                          display=True)# display all predictions
        # scan the plant to detect
        if scanup == True and scan_on == True:
            scanup = system.move_pump_up()

        elif scandown == True and scan_on == True:
            scandown = system.move_pump_down()

        elif scan_on == True:
            system.move_pump_centre('all') # recentre nozzle
            return prediction
    
    system.move_pump_centre('tilt')
    return prediction

# In demo mode rover will fertilizer both sides of the garden row
def fertilize_row(row, cnn):
    global system, SCAN_ON
    # position nozzle Left side
    position = {0:"left", 1:"right"}
    stage = "flowering"
    message = []
    for i in range(2):
        if i == 0:# fertilize left row
            while system.move_pump_left() == True:
                pass
        
        else:# fertilize right row
            while system.move_pump_right() == True:
                pass

        # Detect plant type
        prediction = detect_crop(cnn, SCAN_ON)
        if prediction[1] > 0.70:    
            # Select plant from nutrient database and select growth stage
            plant = cropNutrients.List.get(prediction[0]) 
            # fertilizer amount for selected growth stage
            ml = plant.get(stage)
            msg = pump_liquid(ml) 
            # Log data to Farmbot database
            resp = log_data(plant=prediction[0], liquid=ml, stage="flowering")      
            message.append(" Plant %s on %s column was given %d ml" %(prediction[0], position[i],  ml)) 
            message.append("FarmData Logging... %s"%(resp))

        else:
            message.append("Error! Could not detect crop in row[%d] column[%d]"\
                    %(row,  i))
    # Re-centre pump and update tank level
    message.append(msg)
    system.move_pump_centre('all') 
    return message

# Process user input commands from terminal
def process_user_input(cnn, new_line, sys, current_mode, std):
    msg = new_line.split(' ')
    global system, stream, SCAN_ON
    system = sys
    stream  = std
    # Update value of liquid left in tank 
    if msg[0] == "refill" and is_number(msg[1]):
        TANK_LEVEL =  int(msg[1])
        return "Tank level updated"
    # Clear screen
    elif new_line == "clear":
        stream.clear()
        return ''
    #Check if Mode == Demo Mode and command is valid
    elif current_mode == list_modes.get(DEFAULT_MODE):
        if msg[0] == 'fertilize' and msg[1] == 'row' and is_number(msg[2]):
            return fertilize_row(int(msg[2]), cnn)
        
        else:
            return "Invalid command! Use cmd[fertilize row]"
    #Check if Mode == Pump Mode and command is valid
    elif current_mode == list_modes.get(2):
        if is_number(msg[0]) and msg[1] == "ml":
            return pump_liquid(msg[0])

        else:
            return "Invalid command! Must have formart e.g. [10 ml]"
    # Check if Mode ==  Vision Mode
    else:
        if new_line == "detect":
            predictions = detect_crop(cnn, not(SCAN_ON))
            return  "Predicted label - %s, Score: [%5f]" % (predictions[0], predictions[1])

        else:
            return "Invalid command! Use [detect]"
