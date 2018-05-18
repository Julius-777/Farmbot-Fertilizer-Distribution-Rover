import os
import json
import requests
import time

# Construct JSON Message to send to Farmbot Database

path = 'https://farmbot-data.uqcloud.net/apix/record'

def logging_data(args):
    plant = args.get("plant", default=None)
    liquid_ml = args.get("liquid",default=None)
    current_stage = args.get("stage",default=None)

    ts = int(time.time())
    myUSER = "s4358870"
    stages = {1:"seedling", 2:"growing", 3:"flowering"} # Plant growth stages
    myPASS = "RelativelyDeathLanesAptly"
    req = json.dumps({"userid":myUSER,"passhash":myPASS,
      "timestamp": int(ts *1e3), # time in milliseconds ( need times by 1000)
      "tags":{"location":"indoor"},
      "data":{"fertilizer":liquid_ml,
            "growth_stage":stages[current_stage]}
    })
    resp = requests.post(path, req) # Send JSON string
    print(resp)
    return resp
