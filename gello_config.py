import dynamixel_utils
import os

if os.path.exists("/dev/serial"):
    device_name = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VY5B-if00-port0"
else:
    device_name = "/dev/tty.usbserial-FT94VY5B"
motors = ['XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330']

dm = dynamixel_utils.DynaManager(device_name, motors)

dm.read_motor_positions_rad()