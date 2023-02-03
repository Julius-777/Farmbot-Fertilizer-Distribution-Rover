import Raspi_UI, WebAPI
from image_recognition import CNN_eval as ImageDetection
import time, os

def main():
    terminal = Raspi_UI.UserCmdLine()
    cnn = ImageDetection.PlantDetection()

    while True:
        data = Raspi_UI.process_user_input(terminal, cnn)
        if type(data) is dict:
            WebAPI.logging_data(data) # Send data to farmbot database
        time.sleep(0.2)

if __name__ == "__main__":
    main()
