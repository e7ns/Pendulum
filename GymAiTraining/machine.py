import serial
import time
import math
import re
import logging
import random

#logging.basicConfig(
#    level=logging.DEBUG,  # Set to DEBUG to capture debug messages
#    format='%(asctime)s - %(levelname)s - %(message)s'
#)

ARDUINO_PORT = "COM6"  # Replace with your Arduino port
PRINTER_PORT = "COM4"  # Replace with printer port
BAUD_RATE_ARDUINO = 9600
BAUD_RATE_PRINTER = 115200

FEEDFAST = 25000
LARGEX = 25
MEDX = 15
SMALLX = 5

# X-axis limits (printer range)
MIN_X = 10
MAX_X = 210
POSITION_EPSILON = 0.5

def send_gcode(printer, command, timeout=10):
    if printer:
        try:
            printer.flushInput()  # Clear any existing input
            printer.write(f"{command}\n".encode())
            logging.info(f"Sent: {command}")

            # Wait for 'ok' response or timeout
            start_time = time.time()
            while True:
                if printer.in_waiting > 0:
                    response = printer.readline().decode().strip()
                    logging.info(f"Printer Response: {response}")
                    if response.lower().startswith('ok'):
                        break
                    elif response.lower().startswith('error'):
                        logging.error(f"Printer reported an error: {response}")
                        break
                if time.time() - start_time > timeout:
                    logging.error(f"Timeout waiting for response to command: {command}")
                    break
        except Exception as e:
            logging.error(f"Error sending G-code: {e}")

# Function to get current X position using M114
def get_current_x(printer, timeout=5):
    current_x = None
    if printer:
        try:
            printer.flushInput()
            printer.write(b"M114\n")
            logging.info("Sent: M114")

            start_time = time.time()
            while True:
                if printer.in_waiting > 0:
                    response = printer.readline().decode().strip()
                    logging.info(f"Printer Response: {response}")
                    # Example response: "X:20.00 Y:0.00 Z:0.30 E:0.00 Count X:200 Y:0 Z:0"
                    match = re.search(r'X:([\-0-9.]+)', response)
                    if match:
                        current_x = float(match.group(1))
                        break
                if time.time() - start_time > timeout:
                    logging.error("Timeout waiting for M114 response.")
                    break
                # time.sleep(0.05)  # Polling interval
        except Exception as e:
            logging.error(f"Error getting current position: {e}")
    return current_x

# Function to move to a target X position and wait until movement is complete
def perform_action(printer, action):
    current_x = get_current_x(printer)
    feedrate = FEEDFAST
    new_x=0
    if action == 0:
        new_x = current_x - LARGEX
    elif action == 1:
        new_x = current_x - MEDX
    elif action == 2:
        new_x = current_x - SMALLX
    elif action == 4:
        new_x = current_x + SMALLX
    elif action == 5:
        new_x = current_x + MEDX
    elif action == 6:
        new_x = current_x + LARGEX
    else:
        time.sleep(0.1)
        return

    target_x = min(max(MIN_X, new_x), MAX_X)

    move_to_position(printer, target_x, feedrate)


def move_to_position(printer, target_x, feedrate, timeout=30):
    if printer:
        send_gcode(printer, f"G1 X{target_x:.2f} F{feedrate}")
        # Wait until the printer reaches the target position
        start_time = time.time()
        while True:
            current_x = get_current_x(printer)
            if current_x is not None:
                if abs(current_x - target_x) <= POSITION_EPSILON:
                    logging.info(f"Reached target X: {current_x:.2f}")
                    break
            if time.time() - start_time > timeout:
                logging.error(f"Timeout waiting to reach X={target_x}")
                break

def read_encoder(arduino):
    """
    Reads angle (theta) and angular velocity (omega) from the Arduino's Serial output.

    Expected Serial Output Format:
        "ANGLE:180.00 VELOCITY:0.00"

    Parameters:
        arduino (serial.Serial): The Serial connection to the Arduino.

    Returns:
        tuple:
            theta (float): The current angle in degrees.
            omega (float): The current angular velocity in degrees per second.
    """
    # Default values in case of read failure
    theta = 69.0  # Default angle value if no valid data is read
    omega = 0.0  # Default angular velocity if no valid data is read

    # Regular expression pattern to extract angle and velocity
    pattern = re.compile(r"ANGLE:([-+]?[0-9]*\.?[0-9]+)\s+VELOCITY:([-+]?[0-9]*\.?[0-9]+)")

    while arduino.in_waiting > 0:
        line = ""  # Initialize line to ensure it's always defined
        try:
            # Read the incoming line from Arduino
            line = arduino.readline().decode('utf-8').strip()
            if not line:
                continue  # Skip empty lines

            # Use regular expressions to extract angle and velocity
            match = pattern.match(line)
            if match:
                theta = float(match.group(1))
                omega = float(match.group(2))
            else:
                logging.warning(f"Unexpected line format: '{line}'")
        except UnicodeDecodeError:
            logging.error("Error decoding bytes from Arduino. Ensure the baud rate and encoding are correct.")
        except ValueError:
            logging.error(f"Error converting data to float: '{line}'")
        except Exception as e:
            logging.error(f"Unexpected error while reading encoder data: {e}")
    logging.debug(f"Read theta: {theta:.2f}°, Angular Velocity: {omega:.2f}°/s")
    return theta, omega
def get_state(arduino, printer):
    theta, omega = read_encoder(arduino)
    current_x = get_current_x(printer)
    return theta, omega, current_x
def setup():

    try:
        # Connect to Arduino and printer
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE_ARDUINO, timeout=1)
        logging.info(f"Connected to Arduino on {ARDUINO_PORT} at {BAUD_RATE_ARDUINO} baud.")
        time.sleep(1)  # Allow time for Arduino to reset
        arduino.flushInput()

        printer = serial.Serial(PRINTER_PORT, BAUD_RATE_PRINTER, timeout=2)
        logging.info(f"Connected to printer on {PRINTER_PORT} at {BAUD_RATE_PRINTER} baud.")
        time.sleep(1)  # Allow time for printer to initialize
        printer.flushInput()
        logging.info("Starting pendulum balancing...")
        send_gcode(printer, "G28 X")  # Home X-axis (move to mechanical X0)
        time.sleep(3)
        send_gcode(printer, "G92 X0")  # Set current position to X=0
        move_to_position(printer, 110, feedrate=10000)  # Initialize to X=110
        return arduino, printer

    except Exception as e:
        logging.error(f"Error: {e}")
def calibrate(printer):
    send_gcode(printer, "G28 X")  # Home X-axis (move to mechanical X0)
    time.sleep(3)
    send_gcode(printer, "G92 X0")  # Set current position to X=0
    move_to_position(printer, 110, feedrate=3000)
def perform_reset(arduino, printer):
    move_to_position(printer, 110, feedrate=3000)
    time.sleep(3)
    # theta, omega = read_encoder(arduino)
    # if theta > 180:
    #     move_to_position(printer, 75, feedrate=5000)
    #     move_to_position(printer, 65, feedrate=1000)
    #     move_to_position(printer, 75, feedrate=1000)
    #     move_to_position(printer, 85, feedrate=2000)
    # else:
    #     move_to_position(printer, 155, feedrate=5000)
    #     move_to_position(printer, 165, feedrate=1000)
    #     move_to_position(printer, 155, feedrate=1000)
    #     move_to_position(printer, 145, feedrate=2000)
    # time.sleep(1)
