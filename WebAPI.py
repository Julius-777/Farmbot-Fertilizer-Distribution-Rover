import os
import json
import requests
import time
from flask import Flask
url_path = 'https://farmbot-data.uqcloud.net/apix/record'
url_path2 =
test = {"Plant": "tomato","Tank_Level": 1500}

application = Flask(__name__)
# Map Url to  a return value of function home
@application.route('/', methods=['GET', ' POST'])
def home(): # Home page

    return "Farmbot Fertilizer System"

if __name__ == "__main__":
    application.run(debug=True) #Start this webserver
