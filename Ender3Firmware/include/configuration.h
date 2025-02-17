// Configuration.h

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// Define your motherboard
#define MOTHERBOARD BOARD_MELZI_CREALITY

// Printer Name
#define CUSTOM_MACHINE_NAME "Ender3Pendulum"

// Serial Settings
#define SERIAL_PORT 0
#define BAUDRATE 115200

// Define number of extruders
#define EXTRUDERS 1

// Disable unused axes by setting steps to 0 or minimal values
#define DEFAULT_AXIS_STEPS_PER_UNIT   { 80.00, 0.00, 0.00, 93.00 } // X, Y, Z, E
#define DEFAULT_MAX_FEEDRATE          { 500, 0, 0, 25 }
#define DEFAULT_MAX_ACCELERATION      { 500, 0, 0, 1000 }
#define DEFAULT_ACCELERATION          500
#define DEFAULT_RETRACT_ACCELERATION  500
#define DEFAULT_TRAVEL_ACCELERATION   1000

// Invert stepper directions as per your hardware
#define INVERT_X_DIR false
#define INVERT_Y_DIR false
#define INVERT_Z_DIR false
#define INVERT_E0_DIR true

// Homing settings
#define X_HOME_DIR -1
#define Y_HOME_DIR 0 // Disabled
#define Z_HOME_DIR 0 // Disabled

// Bed size
#define X_BED_SIZE 235
#define Y_BED_SIZE 235
#define Z_BED_SIZE 250

// Enable EEPROM
#define EEPROM_SETTINGS

#endif // CONFIGURATION_H
