// main.cpp
#include <Arduino.h>
#include "Configuration.h" // Include Configuration.h
#include "pins.h"
#include "main.h"

long xPositionMin = 0;
long xPositionMax = 220*80; // 220mm * 80 steps/mm

// State variables
bool isTestRun = true;

bool isXMoving = false;
long xPosition = 1; // Current position in steps

// Function Prototypes
void setupStepper();
void moveX(long steps, float speed = 50000); // Updated to include speed
void stopX();
void handleSerialCommands();


void setup() {
  // Initialize Serial Communication
  USB_SERIAL_PORT.begin(BAUDRATE);
  while (!USB_SERIAL_PORT) {
    ; // Wait for Serial port to connect. Needed for native USB
  }

  pinMode(POWER_SWITCH_PIN, INPUT_PULLUP);

  pinMode(X_MIN_PIN, INPUT_PULLUP);
  pciSetup(X_MIN_PIN);
  
  setupStepper();

  //moveX(110*80); // Move to the center

  USB_SERIAL_PORT.println("Setup complete.");
}

void loop() {
  handleSerialCommands();  
}

// Initialize Stepper Motor Pins
void setupStepper() {
  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(X_ENABLE_PIN, OUTPUT);

  // Set initial direction
  digitalWrite(X_DIR_PIN, LOW); // Reflects INVERT_X_DIR = false

  // Disable stepper initially
  digitalWrite(X_ENABLE_PIN, HIGH); // Disable stepper
  USB_SERIAL_PORT.println("--> Move rod to the left <--");
  delay(3000); 
  digitalWrite(X_ENABLE_PIN, LOW); // Enable stepper

}

// Move X-axis by specified steps and speed
void moveX(long steps, float speed) {
  if (steps == 0) return;

  bool forward = steps > 0;
  isXMoving = true;

  // Set direction
  if (steps > 0) {
    digitalWrite(X_DIR_PIN, LOW); // Based on INVERT_X_DIR in Configuration.h
  } else {
    digitalWrite(X_DIR_PIN, HIGH);
    steps = -steps; // Make steps positive
  }

  // Calculate stepDelay based on speed
  // speed: steps per second
  // stepDelay in microseconds = 1,000,000 / speed
  // Clamp speed to prevent division by zero or extremely high speeds
  if (speed <= 0) speed = 1; // Minimum speed
  if (speed > 50000) speed = 50000; // Maximum speed
  unsigned int stepDelay = 1000000 / speed;

  for (long i = 0; i < steps; i++) {
    
    if (steps < 0 || (xPosition >= xPositionMax && forward) || (xPosition <= xPositionMin && !forward)) {
      stopX();
      break;
    }

    digitalWrite(X_STEP_PIN, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(X_STEP_PIN, LOW);
    delayMicroseconds(stepDelay);

    xPosition += forward ? 1 : -1;
  }

  isXMoving = false;
}

// Stop X-axis movement
void stopX() {  
  isXMoving = false;
}

// Handle incoming serial commands
void handleSerialCommands() {
      if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        long steps;
        float speed;

        //Serial.println(input);

        // Parse the input string
        int firstSpaceIndex = input.indexOf(' ');
        if (firstSpaceIndex != -1) {
            steps = input.substring(0, firstSpaceIndex).toInt();
            speed = input.substring(firstSpaceIndex + 1).toFloat();
        } else {
            // Handle error: input format is incorrect            
            return;
        }

        // Call moveX with the parsed values
        moveX(steps, speed);

        Serial.println(xPosition);
    }
}



