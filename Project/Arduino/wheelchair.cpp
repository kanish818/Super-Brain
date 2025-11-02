#include <Servo.h>

Servo motor;

void setup() {
  Serial.begin(9600);
  motor.attach(9);  
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'F') {
      motor.write(180);  // Move forward
    } else if (command == 'B') {
      motor.write(0);  // Move backward
    }
  }
}
