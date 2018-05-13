import Raspi_UI
import CNN_eval as cnn
import time


def main():
    terminal = Raspi_UI.UserCmdLine()
    while True:
        Raspi_UI.process_user_input(terminal, )
        time.sleep(0.2)

if __name__ == "__main__":
    main()
