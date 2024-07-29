import dynamixel_utils
import os
import numpy as np

if os.path.exists("/dev/serial"):
    device_name = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VY5B-if00-port0"
else:
    device_name = "/dev/tty.usbserial-FT94VY5B"
motors = ['XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330']

dm = dynamixel_utils.DynaManager(device_name, motors)
motor_positions = dm.log_motor_positions()
offsets = []