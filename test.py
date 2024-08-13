import time
import numpy as np
import mujoco
import mujoco.viewer
import dynamixel_utils
import os
import math as m

print(mujoco.__file__)

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

#default config for setup, modify to change reference position (test with single arm in sim)
start_angles = [0, 3*np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, 0]

#gello info

#-1.0334554334554336
offsets = [0, 0.9896214896214897, -0.5402930402930404, 1.49995115995116, 0.00012210012210012692, -1.056166056166056]
joint_orientations = [1, 1, -1, 1, 1, 1]
gripper_open_pos = 1206
gripper_range = 550

start_vals = [
        -4.065179173146383,
        -0.8556114000133057,
        1.419995133076803,
        -3.108495374719137,
        -1.3419583479510706,
        0,
    ]

for id in range(len(offsets)):
  offsets[id] = offsets[id] * np.pi

def init_controller(model, data):
  data.qpos[16] = start_vals[0]
  data.qpos[17] = start_vals[1]
  data.qpos[18] = start_vals[2]
  data.qpos[19] = start_vals[3]
  data.qpos[20] = start_vals[4]
  data.qpos[21] = start_vals[5]

#range of input angles: -pi to pi
def controller(model, data):
  #gello data recording
  data.qpos[16] = start_vals[0]
  data.qpos[17] = start_vals[1]
  data.qpos[18] = start_vals[2]
  data.qpos[19] = start_vals[3]
  data.qpos[20] = start_vals[4]
  data.qpos[21] = start_vals[5]
  print(start_vals)
    
  

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
