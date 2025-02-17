#Put the following line into the terminal to install modules
#pip install pyserial keyboard
#'a'/'d' and LEFT/RIGHT control
#DOES NOT WORK PROPERLY BECAUSE EMERGENCY PARSER IS NOT ENABLED
import serial
import keyboard
import time

# Serial port configuration
SERIAL_PORT = "COM4"  # Replace with the correct port for your printer
BAUD_RATE = 115200    # Common baud rates are 115200 or 250000

# Function to connect to the printer and initialize position
def connect_to_printer():
    try:
        # Establish connection to the printer
        printer = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for the connection to initialize
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

        # Initialize position to X0
        initialize_position(printer)
        return printer

    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        return None

# Function to send G-code commands
def send_gcode(printer, command):
    if printer:
        try:
            printer.write(f"{command}\n".encode())  # Send G-code command
            response = printer.readline().decode().strip()  # Read response
            print(f"Sent: {command} | Response: {response}")
        except Exception as e:
            print(f"Error sending G-code: {e}")

# Function to initialize the printer position to X=0
def initialize_position(printer):
    print("Initializing printer position...")
    send_gcode(printer, "G21")       # Set units to millimeters
    send_gcode(printer, "G28 X")     # Home X-axis (move to mechanical X0)
    time.sleep(2)                    # Wait for homing to complete
    send_gcode(printer, "G92 X0")    # Set current position to X=0
    print("Printer head moved to X=0 and position set to 0.")

# Function for real-time keyboard control
def control_printer(printer):
    print("Use the left/right arrow keys or 'a'/'d' to move the printer in the X-axis.")
    print("Press 'q' to quit.")

    feedrate = 30000  # Movement speed in mm/min
    min_x, max_x = 10, 210  # X-axis limits
    moving_direction = None  # Track current direction: 'left', 'right', or None

    while True:
        if keyboard.is_pressed('left') or keyboard.is_pressed('a'):  # Move left
            if moving_direction != 'left':  # Only send a new command if direction changes
                send_gcode(printer, "M410")  # Emergency stop to clear ongoing motion
                send_gcode(printer, f"G1 X{min_x} F{feedrate}")  # Move infinitely left
                print(f"Moving left towards X={min_x} at {feedrate} mm/min")
                moving_direction = 'left'

        elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):  # Move right
            if moving_direction != 'right':  # Only send a new command if direction changes
                send_gcode(printer, "M410")  # Emergency stop to clear ongoing motion
                send_gcode(printer, f"G1 X{max_x} F{feedrate}")  # Move infinitely right
                print(f"Moving right towards X={max_x} at {feedrate} mm/min")
                moving_direction = 'right'

        elif keyboard.is_pressed('q'):  # Quit program
            send_gcode(printer, "M410")  # Emergency stop before quitting
            print("Exiting control...")
            break

        else:
            if moving_direction is not None:  # Stop only if movement was happening
                send_gcode(printer, "M410")  # Emergency stop
                print("Stopping movement.")
                moving_direction = None

    print("Stopping printer control.")
    send_gcode(printer, "M84")  # Disable motors
    printer.close()
    print("Connection closed.")

# Main program
if __name__ == "__main__":
    printer_connection = connect_to_printer()
    if printer_connection:
        control_printer(printer_connection)
