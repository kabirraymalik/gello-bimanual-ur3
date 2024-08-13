

import rtde_control
import rtde_receive
import rtde_io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from loguru import logger
from typing import List
import time
import numpy as np
import dynamixel_utils, os

class UR3EArm:
    def __init__(self, ip, name):
        self.name = name
        self.ur_c = rtde_control.RTDEControlInterface(ip)
        self.ur_r = rtde_receive.RTDEReceiveInterface(ip)
        self.ur_io = rtde_io.RTDEIOInterface(ip)

        self.default_pos = DEFAULT_UR3E_POS[name]
        self.curr_target = self.get_joint_pos()[1]

        self.ur_c.zeroFtSensor()
        logger.info(f"[UR3EArm] Initialized {name} arm")

    # ====== read sensors ======
    def get_joint_pos(self):
        q = self.ur_r.getActualQ()
        return time.time(), q

    def get_tcp_torque(self):
        torque = np.array(self.ur_r.getActualTCPForce())
        return torque

    def get_joint_torques(self):
        torques = np.array(self.ur_c.getJointTorques())
        return torques

    def get_sensors(self):
        qpos = self.ur_r.getActualQ()
        qvel = self.ur_r.getActualQd()
        tcppos = self.ur_r.getActualTCPPose()
        tcpvel = self.ur_r.getActualTCPSpeed()
        tcpforce = self.ur_r.getActualTCPForce()
        return time.time(), qpos, qvel, tcppos, tcpvel, tcpforce

    # ====== set targets ======
    def update_target(self, action):
        self.curr_target[:] = action

    # ====== command targets ======
    def goto_random_target(self, blocking=True):
        self.curr_target[:] = self.default_pos + np.random.uniform(-0.2, 0.2, 6)
        self.move_joint(self.curr_target, asyn=not blocking)

    def goto_init_target(self, target, blocking=True):
        self.curr_target[:] = target
        self.move_joint(self.curr_target, asyn=not blocking)

    def reset(self, blocking=True):
        self.move_joint(self.default_pos, asyn=not blocking)

    def wait_until_stopped(self):
        while not self.is_stopped():
            time.sleep(1 / 30)
        self.curr_target[:] = self.get_joint_pos()[1]

    def goto_target(self):
        self.servo_joint(self.curr_target)
    
    def restart_forcemode(self):
        self.ur_c.forceModeStop()
        self.ur_c.zeroFtSensor()
        self.ur_c.forceMode([0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], 2, [0.1,0.1,0.1,0.1,0.1,0.1])
    
    def zero_force_sensor(self):
        self.ur_c.zeroFtSensor()

    # ====== ur api ======
    def servo_joint(self, target, time=0.002, lookahead_time=0.15, gain=150):
        # def servo_joint(self, target, time=0.002, lookahead_time=0.1, gain=150):
        """
        Servoj can be used for online realtime control of joint positions.
        It is designed for movements over greater distances.
        time: time where the command is controlling the robot. The function is blocking for time t [S]
        lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time. A low value gives fast reaction, a high value prevents overshoot.
        gain: proportional gain for following target position, range [100,2000]. The higher the gain, the faster reaction the robot will have.
        """
        # cur_torques = np.mean(self.torques, axis=0)
        # correction = self.kp * cur_torques + self.kd * (self.prev_torques - cur_torques)
        # self.prev_torques[:] = cur_torques
        # logger.info(f"[UR3EArm] Servoing to {target} with correction {correction}")
        # self.ur_c.servoJ(target + correction, 0.0, 0.0, time, lookahead_time, gain)
        # self.torques.append(self.get_joint_torques())
        # if len(self.torques) > 50:
        #     del self.torques[:-20]

        self.ur_c.servoJ(target, 0.0, 0.0, time, lookahead_time, gain)

    def move_joint(self, target, interp="joint", vel=1.0, acc=1, asyn=False):
        """
        the movej() command offers you a complete trajectory planning with acceleration, de-acceleration etc.
        It is designed for movements over greater distances.
        """
        if interp == "joint":
            self.ur_c.moveJ(target, vel, acc, asyn)
        elif interp == "tcp":
            self.ur_c.moveL_FK(target, vel, acc, asyn)
        else:
            raise KeyError("interpolation muct be in joint or tcp space")

    def speed_tcp(self, vel, acc=10, t=0):
        """
        Accelerate linearly in tcp space and continue with constant tcp speed.
        """
        self.ur_c.speedL(vel, acc, time=t)

    def speed_joint(self, vel, acc=10):
        """
        Accelerate linearly in joint space and continue with constant joint speed.
        """
        self.ur_c.speedJ(vel, acc)

    # TODO verify the below two are correct
    def move_joint_path(
        self, waypoints: List[np.ndarray], vels, accs, blends, asyn=False
    ):
        """
        waypoints input should be N*6 for joint
        The size of the blend radius is per default a shared value for all the waypoint.
        A smaller value will make the path turn sharper whereas a higher value will make the path smoother.
        """
        assert isinstance(waypoints, np.ndarray), "waypoints must be a numpy array"
        assert waypoints.shape[1] == 6, "dimension of waypoints must be Nx6"
        path = np.hstack(
            (
                waypoints,
                np.array(vels)[..., None],
                np.array(accs)[..., None],
                np.array(blends)[..., None],
            )
        )
        self.ur_c.moveJ(path, asyn)

    def get_joints(self):
        q = self.ur_r.getActualQ()
        return q

    def get_tcp_pose(self):
        # p[:3] is x,y,z
        # p[3:6] is axis angle rotation
        p = self.ur_r.getActualTCPPose()
        return p

    def is_stopped(self):
        return self.ur_c.isSteady()

    def move_until_contact(
        self, vel, thres, acc=0.25, direction=np.array((0, 0, 1, 0, 0, 0))
    ):
        assert len(vel) == 6, "dimension of vel mush be 6"
        self.ur_c.speedL(vel, acceleration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if np.linalg.norm((startforce - force).dot(direction)) > thres:
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def move_linear(self, start, end, vel, thres=17, acc=0.25):
        assert (
            len(start) == 3 and len(end) == 3
        ), "dimension of start and goal must be 3"
        assert (
            len(vel) == 3
        ), "vel is for x,y,z direction and the first nonzero one would be used"

        if abs(end[0] - start[0]) > 0.01:
            direc_x = vel[0]
            direc_y = (
                np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
        elif abs(end[1] - start[1]) > 0.01:
            direc_y = vel[1]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
        else:
            direc_z = vel[2]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )
            direc_y = (
                np.clip((end[1] - start[1]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )

        dire = [direc_x, direc_y, direc_z, 0, 0, 0]
        self.ur_c.speedL(dire, accelaration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if (
                np.linalg.norm((startforce - force).dot(dire)) > thres
                or np.linalg.norm(self.get_pose().translation - end) < 0.01
            ):
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def start_freedrive(self):
        self.ur_c.teachMode()

    def stop_freedrive(self):
        self.ur_c.endTeachMode()

    def stop_script(self):
        self.ur_c.stopScript()

    def reupload_script(self):
        self.ur_c.reuploadScript()
    
    def is_connected(self):
        return self.ur_c.isConnected()
    
    def reconnect(self):
        if self.is_connected():
            return

    def disconnect(self):
        self.ur_c.disconnect()
        self.ur_r.disconnect()
        self.ur_io.disconnect()

    def servo_stop(self):
        self.ur_c.servoStop()

def compare_pos(pos1, pos2, desired_inaccuracy):
    actual_inaccuracy = 0
    for id in range(len(pos1)):
        actual_inaccuracy += abs(pos2[id] - pos1[id])/pos2[id]
    actual_inaccuracy /= len(pos1)
    if actual_inaccuracy < desired_inaccuracy:
        return True
    return False


# play data
DEFAULT_UR3E_POS = {
    "right": np.array([
        -2.198981587086813,
        -2.2018891773619593,
        -1.534730076789856,
        -0.1098826688579102,
        1.2620022296905518,
        0,
    ]),
    "left": np.array([
        -4.065179173146383,
        -0.8556114000133057,
        1.419995133076803,
        -3.108495374719137,
        -1.3419583479510706,
        0,
    ]),
}

## kitchen data
#DEFAULT_UR3E_POS = {
#    "right": np.array([
#        -2.425814930592672,
#        -1.7651893101134242,
#        -2.0178232192993164,
#        -0.08198674142871099,
#        1.2157135009765625,
#        -0.39373714128603154
#    ]),
#    "left": np.array([
#        -3.857370376586914,
#        -1.376403343476369,
#        2.0178232192993164,
#        -3.059605912161082,
#        -1.2157135009765625,
#        0.39373714128603154
#    ]),
#}

if __name__ == "__main__":

    #robot init

    url = UR3EArm("10.42.1.100", "left")
    urr = UR3EArm("10.42.0.100", "right")
    #url = UR3EArm("10.42.1.100", "left")

    urr.move_joint(DEFAULT_UR3E_POS['right'])
    url.move_joint(DEFAULT_UR3E_POS['left'])

    #gello init

    default_gello_pos = [2.1956625578935256, 2.2769833933710637, 2.7909924478045585, 5.976314229905857, 1.4376909970274168, 6.860102932454195]
    default_gello_pos = [2.3184109888030173, 2.364441650394076, 3.0610389958054394, 0.00015343553863686414, 1.8627074390515306, 6.375246630361705]


    #offsets = [0, 0.9896214896214897, -0.5402930402930404, 1.49995115995116, -1, -2.056166056166056]

    urL_angles = [0,0,0,0,0,0]

    #start_angles = [0, 3*np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, 0]
    URL_start_angles = [-4.065179173146383, -0.8556114000133057, 1.419995133076803, -3.108495374719137, -1.3419583479510706, 0]
    prev_pos = URL_start_angles
    joint_orientations = [1, 1, -1, 1, 1, 1]
    gripper_open_pos = 1206
    gripper_range = 550

    #dynamixel setup
    if os.path.exists("/dev/serial"):
        device_name = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VY5B-if00-port0"
    else:
        device_name = "/dev/tty.usbserial-FT94VY5B"
    motors = ['XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330']

    dmR = dynamixel_utils.DynaManager(device_name, motors)

    #config
    base_refresh_rate = 50 #Hz
    runtime = 30 #s
    half_range = 1 #motor angle range limits for safety


    start_time = time.time()
    last_refresh = start_time
    stopped = False
    enable = False


    while not stopped:

        #runtime limit
        if time.time()-start_time>runtime:
            stopped = True

        motor_positions = dmR.log_motor_positions()

        #if 20% avg difference between last pos and curr pos, kill control, return home, and exit loop
        if not compare_pos(motor_positions[:-1], prev_pos, 0.2):
            dmR.enable_control = False
            urL_angles = URL_start_angles
            stopped = True

        #if within 10% avg difference between gello start config and current gello config, start control
        if compare_pos(motor_positions[:-1], default_gello_pos, 0.08):
            dmR.enable_control = True

        if dmR.enable_control: 
            for motor_id in range(len(motor_positions)-1): #only iterating through non-gripper motors
                delta = motor_positions[motor_id] - default_gello_pos[motor_id]
                if joint_orientations[motor_id] == -1: #handling reversed motors
                    delta *= -1
                    urL_angles[motor_id] = URL_start_angles[motor_id] + delta 
                else:
                    urL_angles[motor_id] = URL_start_angles[motor_id] + delta

        print(f'|  URL : {urL_angles} |')
        print(f'| GELLO: {motor_positions} |')
        print('============================================================')

        move = True
        for id in range(len(URL_start_angles)):
            if urL_angles[id] not in range (URL_start_angles[id] - half_range, URL_start_angles[id] + half_range):
                move = False
                break
        if move:
            url.move_joint(urL_angles)

        prev_pos = motor_positions[:-1]

        #time control 
        while (time.time()) < (last_refresh + 1.0/base_refresh_rate):
            time.sleep(0.00001)
            curr_time = time.time()
            time_elapsed = curr_time - last_refresh
            last_refresh = curr_time
        #print(f'refresh rate: {1.0/time_elapsed} Hz')
    
    #return to home when done 
    #urr.move_joint(DEFAULT_UR3E_POS['right'])
    #url.move_joint(DEFAULT_UR3E_POS['left'])