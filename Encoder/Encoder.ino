// Define encoder pins
const int ENCODER_PIN_A = 2;  // Encoder Channel A connected to pin 2
const int ENCODER_PIN_B = 3;  // Encoder Channel B connected to pin 3

volatile long encoderCount = 0;        // Counts encoder steps
const int stepsPerRevolution = 2400;    // Doubled steps per full rotation (adjust if necessary)
float angle = 0.0;                      // Current angle

// Variables for angular velocity calculation
float previousAngle = 0.0;              // Previous angle reading
unsigned long previousMillis = 0;        // Previous time in milliseconds
float angularVelocity = 0.0;             // Angular velocity in degrees per second

// Setup function
void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect. Needed for some boards
  }

  // Initialize encoder pins as inputs with internal pull-up resistors
  pinMode(ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(ENCODER_PIN_B, INPUT_PULLUP);

  // Attach interrupts to both Channel A and Channel B
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), handleEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_B), handleEncoderB, CHANGE);

  // Initialize timing variables
  previousMillis = millis();
  previousAngle = 0.0;
}

// Loop function
void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    // If the command is 'R' (Request), send the latest data
    if (command == 'R') {
      // Safely read the encoder count
      noInterrupts();
      long currentCount = encoderCount;
      interrupts();

      // Calculate current angle
      angle = currentCount * 360.0 / stepsPerRevolution;
      angle = fmod(angle, 360.0);
      if (angle < 0) {
        angle += 360.0;
      }

      // Calculate angular velocity
      float deltaAngle = angle - previousAngle;
      if (deltaAngle > 180.0) deltaAngle -= 360.0;
      else if (deltaAngle < -180.0) deltaAngle += 360.0;

      float deltaTime = (millis() - previousMillis) / 1000.0;
      angularVelocity = deltaAngle / deltaTime;

      previousAngle = angle;
      previousMillis = millis();

      // Send the data
      Serial.println(String(angle, 2) + " " + String(angularVelocity, 2));
    }
  }
  
}

// Interrupt Service Routine for Channel A
void handleEncoderA() {
  bool stateA = digitalRead(ENCODER_PIN_A);
  bool stateB = digitalRead(ENCODER_PIN_B);

  if (stateA == stateB) {
    encoderCount++;
  }
  else {
    encoderCount--;
  }
}

// Interrupt Service Routine for Channel B
void handleEncoderB() {
  bool stateA = digitalRead(ENCODER_PIN_A);
  bool stateB = digitalRead(ENCODER_PIN_B);

  if (stateA != stateB) {
    encoderCount++;
  }
  else {
    encoderCount--;
  }
}