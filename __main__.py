import Raspi_UI
import CNN_eval as cnn
import time, os

# display the stages in plant recognition on terminal
DISPLAY_ON = True
main_path = 'C:\\Users\\jjmiy_000\\Documents\\Github\\' # root directory
# path to directory containing images to evaluate
eval_dataset = os.path.join(main_path, 'eval_dataset')

def main():
    terminal = Raspi_UI.UserCmdLine()
    recognition = cnn.PlantDetection()
    data_generator = recognition.prepare_images(eval_dataset, None)
    predicions = recognition.get_predictions(data_generator, not(DISPLAY_ON))
    #while True:
        #Raspi_UI.process_user_input(terminal, )
        #time.sleep(0.2)

if __name__ == "__main__":
    main()
