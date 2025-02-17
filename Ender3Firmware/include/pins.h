// pins.h

#ifndef PINS_H
#define PINS_H

// Define X-axis Stepper Pins
#define X_STEP_PIN         15
#define X_DIR_PIN          21
#define X_ENABLE_PIN       14

// Defin LCD Pins
#define LCD_PINS_RS     28 // st9720 CS
#define LCD_PINS_ENABLE 17 // st9720 DAT
#define LCD_PINS_D4     30 // st9720 CLK

// Define X-axis Limit Switch
#define X_MIN_PIN          18 // 18

// Define USB Connection (Serial)
#define USB_SERIAL_PORT    Serial

// Define Power Switch (Assuming it's connected to a specific pin)
#define POWER_SWITCH_PIN   2  // Example pin, adjust as necessary

#endif // PINS_H
