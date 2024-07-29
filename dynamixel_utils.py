from dynamixel_sdk import * 
import sys, tty, termios, time
import numpy as np

class DynaManager():
    def __init__(self, device_name, connected_motors):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

        #tracks various motor types in system
        self.motors = connected_motors
        #Dynamixel motor pin info
        xl430_info = {'enable_torque': 64,          'set_position': 116,
                    'set_velocity_P': 78,           'get_position': 132, 
                    'get_velocity': 128,            'get_current': 126,
                    'access_operating_mode': 11,    'set_velocity': 104,
                    'max_velocity': 265,            'max_position': 4095,
                    'baudrate': 57600,              'set_position_velocity': 112,
                    'set_position_P': 84,           'set_position_I': 82,
                    'set_position_D': 80,           'set_velocity_I': 76,
                    'hardware_error_status': 70}
        
        xl330_info = {'enable_torque': 64,          'set_position': 116,
                      'set_current': 102,           'get_position': 132, 
                      'get_velocity': 128,          'get_current': 126,
                      'access_operating_mode': 11,  'set_velocity': 104,
                      'max_velocity': 445,          'max_position': 4095,
                      'baudrate': 57600,            'set_position_velocity': 112,
                      'set_position_P': 84,         'set_position_I': 82,
                      'set_position_D': 80,         'set_velocity_I': 76,
                      'set_velocity_P': 78,         'hardware_error_status': 70}
        
        self.motor_info = {'XL430': xl430_info, 'XL330': xl330_info}

        # DYNAMIXEL Protocol Version (1.0 / 2.0)
        # https://emanual.robotis.com/docs/en/dxl/protocol2/
        PROTOCOL_VERSION            = 2.0

        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            self.getch()
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(int(self.motor_info[self.motors[0]].get('baudrate'))):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            self.getch()
            quit()

    #======================================= UTILS ===========================================#

    def get_motor_info(self, motor_ID, cmd):
        if self.motors[motor_ID-1] in self.motor_info:

            info = self.motor_info[self.motors[motor_ID-1]]
            if cmd in info.keys():
                return info[cmd]
            else:
                print('ERROR: invalid command for selected motor!')
                print(f'motor: {motor_ID} command: {cmd}')
                return -1
        else:
            print(f'ERROR: invalid motor type: {self.motors[motor_ID-1]}')
            return -1

    def getch(self):
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        return ch
        
    #==============================INDIVIDUAL MOTOR FUNCTIONS=================================#

    #================= general
    def read_hardware_status(self, motor_ID):
        dxl_hardware_status, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'hardware_error_status'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            #print(f'motor {motor_ID} hardware status : {dxl_hardware_status}')
            return dxl_hardware_status

    def enable_torque(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info(motor_ID,'enable_torque'), 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} torque on")

    def disable_torque(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info(motor_ID,'enable_torque'), 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} torque off")
    
    #================= operating modes
    
    def set_voltage_mode(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info(motor_ID,'access_operating_mode'), 16)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} mode successfully changed to current")

    def set_position_mode(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info(motor_ID,'access_operating_mode'), 3)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} mode successfully changed to position")

    def set_velocity_mode(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info(motor_ID,'access_operating_mode'), 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} mode successfully changed to velocity")
        
    def set_extended_position_control_mode(self, motor_ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_ID, self.get_motor_info('access_operating_mode'), 4)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {motor_ID} mode successfully changed to current-based position")
    
    def get_operating_mode(self, motor_ID):
        dxl_operating_mode, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'access_operating_mode'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f'Dynamixel {motor_ID} operating mode: {dxl_operating_mode}')
            return dxl_operating_mode

    #================= position control
        
    def set_position(self, motor_ID, position_in_radians):
        if position_in_radians == 0:
            val = 1
        else:
            val = int((self.get_motor_info(motor_ID, 'max_position')* position_in_radians/(2 * np.pi)))
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position'), val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def get_position(self, motor_ID):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'get_position'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_position
    
    def set_position_velocity(self, motor_ID, velocity_as_decimal):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_velocity'), velocity_as_decimal*self.get_motor_info(motor_ID,'set_position_velocity'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    def get_position_velocity(self, motor_ID):
        dxl_present_position_velocity, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_velocity'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_position_velocity
    
    def set_abs_position(self, motor_ID, position_in_ticks):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position'), position_in_ticks)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    def set_position_P(self, motor_ID, value):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_P'), value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else: print(f'Dynamixel {motor_ID} position P value succesfully set to {value}')
    def set_position_I(self, motor_ID, value):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_I'), value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else: print(f'Dynamixel {motor_ID} position I value succesfully set to {value}')
    def set_position_D(self, motor_ID, value):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_D'), value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else: print(f'Dynamixel {motor_ID} position D value succesfully set to {value}')

    def get_position_P(self, motor_ID):
        dxl_present_P, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_P'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_P
    def get_position_I(self, motor_ID):
        dxl_present_P, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_I'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_P
    def get_position_D(self, motor_ID):
        dxl_present_P, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_position_D'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_P
    
    #================= velocity control

    def set_velocity(self, motor_ID, velocity_as_decimal):
        goal_vel = 265.0 * velocity_as_decimal
        if goal_vel <= 265:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_velocity'), int(goal_vel))
            if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    #================= current control

    def set_current(self, motor_ID, current_as_decimal):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'set_current'), current_as_decimal*1750)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def get_current(self, motor_ID):
        dxl_present_current, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, motor_ID, self.get_motor_info(motor_ID,'get_current'))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            return dxl_present_current
    
    #================== helpful other functions

    def read_hardware_status(self):
        output = "|"
        for motor in range(1,len(self.motors)):
            status = self.read_hardware_status(motor)
            output = output + f" motor {motor}: {status}"
        print(output)
        return output
    
    def read_motor_positions(self):
        output = "|"
        for motor in range(1,len(self.motors)):
            pos = self.get_position(motor)
            output = output + f" motor {motor}: {pos} |"
        print(output)
        return output

    def read_motor_positions_rad(self):
            output = "|"
            for motor in range(1,len(self.motors)):
                pos = (self.get_position(motor)/4095)*2*np.pi
                output = output + f" motor {motor}: {pos} rad |"
            print(output)
            return output
    
    def log_motor_positions(self):
        out = []
        for motor in range(1, len(self.motors)):
            pos = (self.get_position(motor)/4095)*2*np.pi
            out.append(pos)
        return out