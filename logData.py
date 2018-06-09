import os
import json
import requests
import time

grafana_path = 'https://farmbot-data.uqcloud.net/apix/record' 

# Construct JSON Message to send to Farmbot Database
def logging_data(**args):
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
    print(req)
    # Send data to farmbot
    resp = requests.post(grafana_path, req) # Send JSON string
    return req
def main():
    for i in range(100):
        logging_data(plant=Tomato, liquid=50, stage="flowering")
        time.sleep(10)
        logging_data(plant=Broccoli, liquid=45, stage="growth")
        logging_data(plant=Cabbage, liquid=100, stage="flowering")
        logging_data(plant=Tomato, liquid=50, stage="flowering")
        logging_data(plant=Tomato, liquid=50, stage="flowering")
        logging_data(plant=Tomato, liquid=50, stage="flowering")

        
