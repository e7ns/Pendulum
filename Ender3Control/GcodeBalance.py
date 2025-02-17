import serial
import time
import math
import re
import logging
# AI will go into
# balance_pendulum(arduino, printer)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture debug messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Serial port configurations
ARDUINO_PORT = "COM5"  # Replace with your Arduino port
PRINTER_PORT = "COM6"  # Replace with printer port
BAUD_RATE_ARDUINO = 115200
BAUD_RATE_PRINTER = 115200
ANGLEMULT = 300
OMEGAMULT = 0
OFFSET = 0
ANGLEOFFSET = 0
FEEDRATE = 20000
ACCELERATION = 20000

# X-axis limits (printer range)
MIN_X = 10
MAX_X = 210

# Epsilon for position comparison
POSITION_EPSILON = 0.5  # mm

# Define the acceptable range around 180 degrees
VERTICAL_MIN = 179.5
VERTICAL_MAX = 180.5


# Function to read angle from Arduino
def read_encoder(arduino, timeout=1.0):
    """
    Sends a request to the Arduino to retrieve the latest angle and angular velocity.

    Expected Serial Output Format:
        "ANGLE:180.00 VELOCITY:0.00"

    Parameters:
        arduino (serial.Serial): The Serial connection to the Arduino.
        timeout (float): Maximum time to wait for a response in seconds.

    Returns:
        tuple:
            theta (float): The current angle in degrees.
            omega (float): The current angular velocity in degrees per second.
    """
    # Default values in case of read failure
    theta = 0.0  # Default angle value if no valid data is read
    omega = 0.0  # Default angular velocity if no valid data is read

    # Regular expression pattern to extract angle and velocity
    pattern = re.compile(r"([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)")

    try:
        # Clear the input buffer to remove any old or incomplete data
        arduino.reset_input_buffer()

        # Send the 'R' command to request data
        arduino.write(b'R')
        # logging.debug("Sent 'R' command to Arduino.")

        # Record the start time to implement timeout
        start_time = time.time()

        while True:
            # Check if data is available to read
            if arduino.in_waiting > 0:
                # Read the incoming line from Arduino
                line_bytes = arduino.readline()
                try:
                    line = line_bytes.decode('utf-8').strip()
                    # logging.debug(f"Received line from Arduino: '{line}'")

                    if not line:
                        continue  # Skip empty lines

                    # Use regular expressions to extract angle and velocity
                    match = pattern.match(line)
                    if match:
                        theta = float(match.group(1))
                        omega = float(match.group(2))
                        logging.info(f"Angle: {theta:.2f}°, Angular Velocity: {omega:.2f}°/s")
                        return theta, omega
                    else:
                        logging.warning(f"Unexpected line format: '{line}'")
                except UnicodeDecodeError:
                    logging.error("Error decoding bytes from Arduino. Ensure the baud rate and encoding are correct.")
                except ValueError:
                    logging.error(f"Error converting data to float: '{line}'")
                except Exception as e:
                    logging.error(f"Unexpected error while parsing data: {e}")

            # Check for timeout
            if (time.time() - start_time) > timeout:
                logging.error("Timeout: No response received from Arduino.")
                break

            # Small sleep to prevent CPU overuse
            time.sleep(0.001)

    except serial.SerialException as e:
        logging.error(f"Serial communication error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    logging.debug(f"Returning default values - Angle: {theta}°, Angular Velocity: {omega}°/s")
    return theta, omega


# Function to send G-code commands and wait for 'ok'
def send_gcode(printer, command, timeout=10):
    if printer:
        try:
            printer.flushInput()  # Clear any existing input
            printer.write(f"{command}\n".encode())
            # logging.info(f"Sent: {command}")

            # Wait for 'ok' response or timeout
            start_time = time.time()
            while True:
                if printer.in_waiting > 0:
                    response = printer.readline().decode().strip()
                    # logging.info(f"Printer Response: {response}")
                    if response.lower().startswith('ok'):
                        # logging.info(f"Printer reported an ok: {response}")
                        break
                    elif response.lower().startswith('error'):
                        logging.error(f"Printer reported an error: {response}")
                        break
                if time.time() - start_time > timeout:
                    logging.error(f"Timeout waiting for response to command: {command}")
                    break
        except Exception as e:
            logging.error(f"Error sending G-code: {e}")

def move_x(steps, speed=50000):
    timeout = 10
    if printer:
        try:
            printer.flushInput()  # Clear any existing input
            command = f"{steps} {speed}\n"
            printer.write(f"{command}\n".encode())
            # logging.info(f"Sent: {command}")

            # Wait for 'ok' response or timeout
            start_time = time.time()
            while True:
                if printer.in_waiting > 0:
                    response = printer.readline().decode().strip()
                    logging.info(f"x Position: {response}")
                    break
                if time.time() - start_time > timeout:
                    logging.error(f"Timeout: {command}")
                    break
        except Exception as e:
            logging.error(f"Error sending command: {e}")

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
                # time.sleep(0.02)  # Polling interval
        except Exception as e:
            logging.error(f"Error getting current position: {e}")
    return current_x


# Function to move to a target X position and wait until movement is complete
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
            time.sleep(0.01)

# Balancing loop
def balance_pendulum(arduino, printer):
    logging.info("Starting pendulum balancing...")
    send_gcode(printer, "G28 X")  # Home X-axis (move to mechanical X0)
    time.sleep(3)
    send_gcode(printer, "G92 X0")  # Set current position to X=0
    move_to_position(printer, 110, feedrate=10000)  # Initialize to X=110
    x_position = 110  # Start at the middle of the range
    send_gcode(printer, f"M204 S{ACCELERATION}")

    # Wait for the stick to be moved to the vertical position (theta ≈ 180.0)
    logging.info("Waiting for stick to be moved to vertical position...")
    while True:
        theta, omega = read_encoder(arduino)
        if VERTICAL_MIN <= theta <= VERTICAL_MAX:
            logging.info("Stick is vertical. Starting balancing loop...")
            break
        logging.debug(f"Current theta: {theta:.2f}")
        time.sleep(0.02)  # Prevent tight loop

    # Start balancing loop only after encoder reads approximately 180
    while True:
        try:
            # Read angle and angular velocity from Arduino
            theta, omega = read_encoder(arduino)

            # Calculate change in position with both angle and angular velocity
            # x_change = (math.sin(math.radians(theta)) * ANGLEMULT) + (omega * OMEGAMULT)
            x_change = (math.sin(math.radians(theta+ANGLEOFFSET)) * ANGLEMULT)

            # Determine new position with offset based on direction
            if x_change > 0.0:
                new_x = 1.0 * x_position + x_change + OFFSET
            elif x_change < 0.0:
                new_x = 1.0 * x_position + x_change - OFFSET
            else:
                new_x = 1.0 * x_position  # No change if x_change is zero

            # Clamp to valid range
            new_x = min(max(MIN_X, new_x), MAX_X)

            # Move to new position and wait until movement is complete
            move_to_position(printer, new_x, feedrate=FEEDRATE)

            # Update current position
            x_position = new_x

            logging.info(f"Theta: {theta:.2f} | X: {x_position:.2f} | Displacement: {x_change:.2f}")

        except KeyboardInterrupt:
            logging.info("Stopping balancing...")
            send_gcode(printer, "M410")  # Emergency stop
            break

    send_gcode(printer, "M84")  # Disable motors
    printer.close()
    arduino.close()
    logging.info("Connections closed.")


# Swing up pendulum
def swing(printer):
    logging.info("Starting pendulum swing...")
    send_gcode(printer, "G28 X")  # Home X-axis (move to mechanical X0)
    time.sleep(3)
    send_gcode(printer, "G92 X0")  # Set current position to X=0
    move_to_position(printer, 50, feedrate=1000)
    time.sleep(0.5)
    move_to_position(printer, 180, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 50, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 180, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 50, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 180, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 50, feedrate=19000)
    time.sleep(0.5)
    move_to_position(printer, 180, feedrate=19000)
    time.sleep(0.5)

    send_gcode(printer, "M84")  # Disable motors
    printer.close()
    arduino.close()
    logging.info("Connections closed.")

def test(arduino, printer):
    logging.info("Starting pendulum balancing...")
    send_gcode(printer, "G28 X")  # Home X-axis (move to mechanical X0)
    time.sleep(3)
    send_gcode(printer, "G92 X0")  # Set current position to X=0
    move_to_position(printer, 110, feedrate=10000)  # Initialize to X=110
    x_position = 110  # Start at the middle of the range
    send_gcode(printer, f"M204 S{ACCELERATION}")
    get_current_x(printer)


# Main program
if __name__ == "__main__":
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

        # while True:
        #    move_x("400 10000")
        #    time.sleep(0.02)
        #    move_x("-400 10000")
        #    time.sleep(0.02)

        # test(arduino, printer)

        #while True:
        #    if printer.in_waiting > 0:
        #        response = printer.readline().decode().strip()
        #        logging.info(f"Printer Response: {response}")
        #        time.sleep(0.1)

        # while True:
        #    read_encoder(arduino)
        #    time.sleep(0.1)
            # read_encoder(arduino)
            # get_current_x(printer)
            # time.sleep(1)

        # balance_pendulum(arduino, printer)
        # swing(printer)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if arduino.is_open:
            arduino.close()
            logging.info("Arduino connection closed.")
        if printer.is_open:
            printer.close()
            logging.info("Printer connection closed.")
