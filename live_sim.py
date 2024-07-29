import time
import numpy as np
import mujoco
import mujoco.viewer

print(mujoco.__file__)


m = mujoco.MjModel.from_xml_path('resources/scene.xml')
d = mujoco.MjData(m)
motorAngles = []

#this is old
def key_listener(keycode):
    #print(keycode)
    if chr(keycode) == ' ':
        global paused
        paused = not paused
    if chr(keycode) == '256':
       endSim = True

def init_controller(model, data):
   pass

#range of input angles: -pi to pi
def controller(model, data):
   data.qpos[0] = 0
   data.ctrl[1] = 15
   data.ctrl[2] = 15

   #print output
   print(f"{data.qpos[0]} | {data.qpos[1]} | {data.qpos[2]} | {data.qpos[3]}")

init_controller(m, d)
mujoco.set_mjcb_control(controller)

paused = False

with mujoco.viewer.launch_passive(m, d, key_callback=key_listener) as viewer:
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
