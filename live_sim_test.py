import time
import numpy as np
import mujoco
import mujoco.viewer
import dynamixel_utils
import os
import math as m

print(mujoco.__file__)

def compare_pos(pos1, pos2, desired_inaccuracy):
    actual_inaccuracy = 0
    for id in range(len(pos1)):
        actual_inaccuracy += abs(pos2[id] - pos1[id])/pos2[id]
    actual_inaccuracy /= len(pos1)
    if actual_inaccuracy < desired_inaccuracy:
        return True
    return False

#dynamixel setup
if os.path.exists("/dev/serial"):
    device_name = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VY5B-if00-port0"
else:
    device_name = "/dev/tty.usbserial-FT94VY5B"
motors = ['XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330', 'XL330']

dmR = dynamixel_utils.DynaManager(device_name, motors)

#mujoco setup
m = mujoco.MjModel.from_xml_path('resources/bimanual-ur3/scene.xml')
d = mujoco.MjData(m)

default_gello_pos = [2.1956625578935256, 2.2769833933710637, 2.7909924478045585, 5.976314229905857, 1.4376909970274168, 6.860102932454195]
default_gello_pos = [2.3184109888030173, 2.364441650394076, 3.0610389958054394, 0.00015343553863686414, 1.8627074390515306, 6.375246630361705]

#offsets = [0, 0.9896214896214897, -0.5402930402930404, 1.49995115995116, -1, -2.056166056166056]

#start_angles = [0, 3*np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, 0]
URL_start_angles = [-4.065179173146383, -0.8556114000133057, 1.419995133076803, -3.108495374719137, -1.3419583479510706, 0]
joint_orientations = [1, 1, -1, 1, 1, 1]
gripper_open_pos = 1206
gripper_range = 550


def init_controller(model, data):
  data.qpos[16] = URL_start_angles[0]
  data.qpos[17] = URL_start_angles[1]
  data.qpos[18] = URL_start_angles[2]
  data.qpos[19] = URL_start_angles[3]
  data.qpos[20] = URL_start_angles[4]
  data.qpos[21] = URL_start_angles[5]

#range of input angles: -pi to pi
def controller(model, data):

  urL_angles = [-4.065179173146383, -0.8556114000133057, 1.419995133076803, -3.108495374719137, -1.3419583479510706, 0]
  motor_positions = dmR.log_motor_positions()
  offset = 16 #in array positions to control the right arm instead of the left
  
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
  
  #right hand
  trigger_pos = dmR.get_position(7)

  for id in range(len(urL_angles)):
    data.qpos[id + offset] = urL_angles[id]

  data.qpos[22] = ability_macro(trigger_pos)
  data.qpos[23] = ability_macro(trigger_pos)
  data.qpos[24] = ability_macro(trigger_pos)
  data.qpos[25] = ability_macro(trigger_pos)
  data.qpos[26] = ability_macro(trigger_pos)
  data.qpos[27] = ability_macro(trigger_pos)
  data.qpos[28] = ability_macro(trigger_pos)
  data.qpos[29] = ability_macro(trigger_pos)
  data.qpos[30] = ability_macro(trigger_pos)
  data.qpos[31] = 2*ability_macro(trigger_pos)

  print(f'UR positions: {urL_angles}')
  

#print output   
  #print(f"sim thetas: {data.qpos[0]} | {data.qpos[1]} | {data.qpos[2]} | {data.qpos[3]} | {data.qpos[4]} | {data.qpos[5]}")
   
def ability_macro(gripper_pos):
  #print(((gripper_open_pos-gripper_pos)/gripper_range))
  return (np.pi/2)*((gripper_open_pos-gripper_pos)/gripper_range)


init_controller(m, d)
mujoco.set_mjcb_control(controller)

paused = False

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    if not paused:
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
