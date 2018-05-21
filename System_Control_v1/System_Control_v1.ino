/*
  Example raspiCmd Serial Passthrough Sketch
  by: Julius Miyumo
  Farmbot Theisis
  date: 19 April 2018

  Uses RN-42 raspiCmd module to
  communicate at 9600 bps (from 115200), and passes any serial
  data between Serial Monitor and raspiCmd module.
*/
#include <Adafruit_MotorShield.h>
#include <stdio.h>
#include <string.h>
#include <SoftwareSerial.h>
#include <Servo.h>

/* execution message commands*/
#define LEFT 0
#define RIGHT 1
#define OTHER 5

// Create the motor shield object with the default I2C address
Adafruit_MotorShield MSHIELD = Adafruit_MotorShield();
Adafruit_DCMotor *leftWheel = MSHIELD.getMotor(3); // Connect a DC motor to port M3
Adafruit_DCMotor *rightWheel = MSHIELD.getMotor(2); // Connect a DC motor to port M2
Adafruit_DCMotor *fert_pump = MSHIELD.getMotor(1); // Connect a DC motor to port M1

Servo servoPan, servoTilt;  // create servo object to control a servo

String data;
int roverSpeed;
int pumpSpeed;

/**
 * Function tells rover to move FORWARD or BACKWARD 
 * for a certain distance
 */
void moveRover(int direc, int x_dis)
{
  if (direc == FORWARD || direc == BACKWARD) 
  {
    int duration = (x_dis*3333)/(roverSpeed); //milliseconds
    leftWheel->run(direc);
    rightWheel->run(direc);
    delay(duration);
    leftWheel->run(RELEASE);
    rightWheel->run(RELEASE);
  } 
}

/**
 * Function tells rover to TURN 90 Deg left or right
 */
void turnRover(char *direc)
{
  char buff[6];
  int x_dis = 5; // cm
  int duration = (x_dis*3333)/(roverSpeed);
  if(!strncmp(direc,"right", strlen("right")))
  {
    rightWheel->run(FORWARD);
    leftWheel->run(BACKWARD);
    Serial.println("Right");
    delay(duration);
    rightWheel->run(RELEASE);
    leftWheel->run(RELEASE);
  } else if(!strncmp(direc,"left", strlen("left"))) 
  {
    rightWheel->run(BACKWARD);
    leftWheel->run(FORWARD);
    Serial.println("Left");
    delay(400);
    leftWheel->run(RELEASE);
    rightWheel->run(RELEASE);
  }  
}

/**
 * Liquid is pumped at 6.67ml/s 
 */
void pumpLiquid(char *message, int quantity)
{ 
  unsigned long duration = (unsigned long)(1000*(float)quantity/6.667); //milliseconds

  if(!strcmp(message,"fertilize"))
  {
    fert_pump->run(FORWARD);
    Serial.print("FERT for ");
    Serial.println(duration);
    delay(duration);
    Serial.println("Done..");
    fert_pump->run(RELEASE);
  }
}

bool isNumber(String number){
  bool c;
  int i;
  for(i=0; i <number.length(); i++)
  {
    c = isDigit(number[i]);
  }
  return c;
}

/**
 * Function points the pump and pi camera 
 * in the chosen direction
 */
void pumpPanTilt(int pan, int tilt)
{
  if ((pan >= 0) && (pan <= 180))
  {
    servoPan.write(pan);
  }

  if ((tilt >= 0) && (tilt <= 180))
  {
    servoTilt.write(tilt);
  }
}

bool executeCommand(String message)
{   char *type, *cmd, *param1, *param2, charBuffer[50];
    int direc, value1, value2;
    message.toCharArray(charBuffer, 50);
    type = strtok(charBuffer,":, ");//Parse data and return information
    cmd = strtok(NULL,":, ");
    param1 = strtok(NULL,":, ");
    param2 = strtok(NULL,":, ");
    
  if (!strcmp(type,"Move"))
  {  // Orientation of pump has been switched on rover
    !strcmp(cmd,"forward")? direc=BACKWARD : 
      !strcmp(cmd,"backward")? direc=FORWARD : direc=OTHER;

    if (direc == OTHER) 
    {
      if (!strcmp(cmd,"turn"))
      {
        // Turn rover left or right 90 deg
        turnRover(param1); 
        return true; // command was valid
      } 
      return false; // command was invalid    
    }
    // move rover forward or backwards in a straight line
    moveRover(direc, String(param1).toInt()); 
    
   } else if (!strcmp(type,"Pump"))
   {
      // Pump fertilizer liquid (ml)
      pumpLiquid(cmd, String(param1).toInt());
      
  } else if (!strcmp(type,"PanTilt") && !strcmp(cmd,"angle"))
  { // Set orientation of pump and camera
    value1 = String(param1).toInt();
    value2 = String(param2).toInt();
    pumpPanTilt(value1, value2);    
    
  } else 
  {
    return false; 
  }
  return true;  
}

void blinkLed()
{
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(5);                       // wait for a second
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(5);                       // wait for a second
}

void setup()
{
  Serial.begin(9600);  // Begin the serial monitor at 9600bps
  //Serial.setTimeout(100) ;
  servoPan.attach(10);  // attaches the servo on pin 9 to the servoPan object
  servoTilt.attach(9);  // attaches the servo on pin 9 to the servoTilt object
  servoPan.write(60);  // Default angle 60 [Range: 0 - 140]
  servoTilt.write(150); // Default angle 150 [Range: 110 - 175]
  
  roverSpeed = 40;  // Set rover speed 0 (off) to 255 (max speed)
  pumpSpeed = 255;  // Max pump (DC 12V)
  
  MSHIELD.begin();  // default f=1.6KHz
  leftWheel->setSpeed(roverSpeed);
  rightWheel->setSpeed(roverSpeed);
  fert_pump->setSpeed(pumpSpeed);
 
  pinMode(LED_BUILTIN, OUTPUT); 
}

void loop()
{
  if (Serial.available()) // If serial monitor used
  {
    /* Incoming Message Formart: */
    //   Exg PanTilt:angle:10,10
    //   Exg Pump:fertilize:20 ml
    //   Exg Move:forward:10 cm or turn:left or turn:right
    String message = Serial.readStringUntil(';');//read line message
    int success = executeCommand(message);
    Serial.println(success);
   }
   blinkLed();
}




