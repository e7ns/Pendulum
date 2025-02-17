import machine
import time
arduino, printer = machine.setup()
time.sleep(10)
machine.perform_reset(arduino, printer)
time.sleep(10)
machine.perform_reset(arduino, printer)