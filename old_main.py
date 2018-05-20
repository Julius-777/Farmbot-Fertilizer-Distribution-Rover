 #!/usr/bin/python
import Raspi_UI
import CNN_eval as ImageDetection
import time, os

def main():
    terminal = Raspi_UI.UserCmdLine()
    cnn = ImageDetection.PlantDetection()

    while True:
        RaspiUI.process_user_input(terminal, cnn)
        time.sleep(0.2)

if __name__ == "__main__":
    main()
