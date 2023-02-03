# Fertilizer Distribution Rover
This project was created for my Electrical Engineering Undergraduate Honours Thesis at The University of Queensland which recieved a distinction. The aim of the project was to designed and build a semi-autonomous fertilizer distribution rover as an extension of the [farmbot](https://farm.bot/) open source project. Feel free to view my [Thesis Report](https://drive.google.com/file/d/1nP5PC57npGuRtNQ3U67S4gIUgEPupZ8T/view?usp=sharing).
 
 ## Summary
 The following technologies were used:
*	Linux shell scripting was used to setup the environment on RasPi to run the python scripts controlling the rover
* Keras and TensorFlow were used to train CNN Deep Learning Networks for Image recognition to identify plants correctly before distributing the fertilizer
* OpenCV was used for preprocessing and transforming image data into Numpy arrays that could be analyzed by the Image recognition Deep Learning Model
* RasPi was interfaced with Arduino microcontroller running C++ code controlling the hardware modules e.g. motors, pumps, wifi.

 ## License
This project is licensed under the MIT License 
