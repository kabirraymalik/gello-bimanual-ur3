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
motorAngles = []
#-1.0334554334554336
offsets = [0, 0.9896214896214897, -0.5402930402930404, 1.49995115995116, 0.00012210012210012692, -2.056166056166056]
joint_orientations = [1, 1, -1, 1, 1, 1]
gripper_open_pos = 1206
gripper_range = 550

for id in range(len(offsets)):
  offsets[id] = offsets[id] * np.pi
  

def init_controller(model, data):
  pass
  #data.qpos[0] = start_angles[0]
  #data.qpos[1] = start_angles[1]
  #data.qpos[2] = start_angles[2]
  #data.qpos[3] = start_angles[3]
  #data.qpos[4] = start_angles[4]
  #data.qpos[5] = start_angles[5]

#range of input angles: -pi to pi
def controller(model, data):
  #gello data recording

  #right arm
  offset = 16
  motor_positions = dmR.log_motor_positions()
  for motor_id in range(len(motor_positions)-1):
    if joint_orientations[motor_id] == -1:
        val = motor_positions[motor_id] + offsets[motor_id]
        difference = start_angles[motor_id] - val
        data.qpos[motor_id + offset] = start_angles[motor_id] + difference
    else:
      data.qpos[motor_id + offset] = motor_positions[motor_id] + offsets[motor_id]
  
  #right hand
  trigger_pos = dmR.get_position(7)
  data.qpos[21] = ability_macro(trigger_pos)
  data.qpos[22] = ability_macro(trigger_pos)
  data.qpos[23] = ability_macro(trigger_pos)
  data.qpos[24] = ability_macro(trigger_pos)
  data.qpos[25] = ability_macro(trigger_pos)
  data.qpos[26] = ability_macro(trigger_pos)
  data.qpos[27] = ability_macro(trigger_pos)
  data.qpos[28] = ability_macro(trigger_pos)
  data.qpos[29] = ability_macro(trigger_pos)
  data.qpos[30] = ability_macro(trigger_pos)

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
        
        print(f'position motor 6: {dmR.get_position(6)}')

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
