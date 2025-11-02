#include <SPI.h>

const int eegPinC3 = A0;
const int eegPinCz = A1;
const int eegPinC4 = A2;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int eegSignalC3 = analogRead(eegPinC3);
  int eegSignalCz = analogRead(eegPinCz);
  int eegSignalC4 = analogRead(eegPinC4);
  
  Serial.print(eegSignalC3);
  Serial.print(",");
  Serial.print(eegSignalCz);
  Serial.print(",");
  Serial.println(eegSignalC4);
  delay(10);  
}
