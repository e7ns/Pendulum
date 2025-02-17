// ISR for Limit Switch (if using interrupt)
// volatile bool limitSwitchTriggered = false;
// volatile unsigned long limitSwitchTriggeredTime = 0;

//ISR(PCINT1_vect) { // PCINT1_vect corresponds to Port C
  // Check if X_MIN_PIN is LOW (assuming active LOW for the limit switch)
  //if (!(PINC & _BV(2))) { // Check if PC2 is LOW
    //limitSwitchTriggered = true;
    // Optionally, record the trigger time
    //limitSwitchTriggeredTime = millis(); // Not recommended in ISR
  //}

  // if (digitalRead(X_MIN_PIN) == LOW) {
  //   limitSwitchTriggered = true;
  //   // Optionally, record the trigger time
  //   limitSwitchTriggeredTime = millis();
  // }
//}


// void limitSwitchISR() {
//   limitSwitchTriggered = true;
//   //limitSwitchTriggeredTime = millis();  
// }


  //USB_SERIAL_PORT.println(digitalPinToInterrupt(X_MIN_PIN));
  //attachInterrupt(digitalPinToInterrupt(X_MIN_PIN), limitSwitchISR, FALLING); // Trigger on falling edge FALLING


// void TestRun(){
//   if (isTestRun) {
    
//     USB_SERIAL_PORT.println("Move 1");
//     moveX(4000);
//     delay(1000);

//     USB_SERIAL_PORT.println("Move 2");
//     moveX(-4000);
//     delay(1000);

//     // USB_SERIAL_PORT.print("Stop");
//     // stopX();
//     // delay(1000);

//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     moveX(4000);
//     delay(20);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     moveX(-4000);
//     delay(20);
//     moveX(4000);
//     delay(20);
//     moveX(-4000);
//     delay(20);
//     moveX(4000);
//     delay(20);
//     moveX(-4000);
//     delay(20);
//     moveX(4000);
//     delay(20);
//     moveX(-4000);
//     delay(20);
//     moveX(4000);    

//     // Print current X position
//     USB_SERIAL_PORT.print("Current X position: ");
//     USB_SERIAL_PORT.println(xPosition);
//     delay(1000);

//     isTestRun = false;
//   }
// }