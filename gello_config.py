import dynamixel_utils
import os
import numpy as np
import time

if os.path.exists("/dev/serial"):
    device_name = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VY5B-if00-port0"
else:
    device_name = "/dev/tty.usbserial-FT94VY5B"
motors = ['XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330']

dm = dynamixel_utils.DynaManager(device_name, motors)
offsets = []

#starting motor offsets should be: -pi, pi, -pi/2, 3pi/2, 3pi/2, 2pi

start_angles = [0, 3*np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, 0]

print('place gello in reference position (trigger open)')
input = input('ready?\n')
if input == 'y':
    motor_positions = dm.log_motor_positions()
    for i in range(len(start_angles)):
        offsets.append(start_angles[i] - motor_positions[i])
    print(f'offsets: {offsets}')
    in_pi = []
    for val in offsets:
        in_pi.append(val/np.pi)
    print(f'in pi: offsets = {in_pi}')
    print(f'gripper angle: {dm.get_position(7)}')


