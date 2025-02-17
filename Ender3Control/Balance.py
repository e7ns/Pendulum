import serial
import time
import math
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture debug messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Serial port configurations
ARDUINO_PORT = "COM5"
PRINTER_PORT = "COM6"
BAUD_RATE_ARDUINO = 115200
BAUD_RATE_PRINTER = 115200
ROD_LENGTH = 300
LEAN_FORWARD_X = 110
LEAN_BACKWARD_X = 110
LEAN_ANGLE = 0.005
MOVE_ACTION_LIMIT = 50

# X-axis limits (printer range)
MIN_X = 0.0
MAX_X = 220.0

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


def move_x(steps, speed=50000):
    steps = int(steps)
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


# Balancing loop
def balance_pendulum(arduino, printer):

    logging.info("Starting pendulum balancing...")
    time.sleep(3)

    x_position = 110.0
    move_x(x_position * 80, 20000)

    # Wait for the stick to be moved to the vertical position (theta ≈ 180.0)
    logging.info("Waiting for stick to be moved to vertical position...")
    while True:
        theta, omega = read_encoder(arduino)
        if VERTICAL_MIN <= theta <= VERTICAL_MAX:
            logging.info("Stick is vertical. Starting balancing loop...")
            break

        time.sleep(0.02)  # Prevent tight loop

    # Start balancing loop only after encoder reads approximately 180
    while True:
        try:
            # Read angle and angular velocity from Arduino
            theta, omega = read_encoder(arduino)

            lean = 0.0
            if x_position < LEAN_FORWARD_X:
                lean = LEAN_ANGLE * (LEAN_FORWARD_X - x_position)
            elif x_position > LEAN_BACKWARD_X:
                lean = -LEAN_ANGLE * (x_position - LEAN_BACKWARD_X)

            # Calculate change in position with both angle and angular velocity
            x_change = (math.sin(math.radians(theta + lean)) * ROD_LENGTH)

            x_move_to_position_compensate = ROD_LENGTH * (omega * math.pi / 360) * -0.04
            x_change = x_change + x_move_to_position_compensate

            # x_change = x_change * max(math.log2(abs(x_change)), 1)

            x_change = min(max(x_change, -MOVE_ACTION_LIMIT), MOVE_ACTION_LIMIT)

            new_x = x_position + x_change

            # Clamp to valid range
            new_x = min(max(MIN_X, new_x), MAX_X)

            # Move to new position and wait until movement is complete
            move_x(x_change * 80, 30000)

            # Update current position
            x_position = new_x

            logging.info(f"Theta: {theta:.4f} | Omega: {omega:.2f} | X: {x_position:.2f} | X_Theta: {x_change:.2f}")

        except KeyboardInterrupt:
            logging.info("Stopping balancing...")
            break

    printer.close()
    arduino.close()
    logging.info("Connections closed.")


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

        # while True:
        #    if printer.in_waiting > 0:
        #        response = printer.readline().decode().strip()
        #        logging.info(f"Printer Response: {response}")
        #        time.sleep(0.1)

        # while True:
        #   read_encoder(arduino)
        #   time.sleep(0.1)

        balance_pendulum(arduino, printer)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if arduino.is_open:
            arduino.close()
            logging.info("Arduino connection closed.")
        if printer.is_open:
            printer.close()
            logging.info("Printer connection closed.")
