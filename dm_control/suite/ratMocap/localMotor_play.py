#localMotor_play.py

# Copyright 2019 Jeff Rhoades.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import progressbar

from absl import app
from absl import flags
from pathlib import Path
import os
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers

import scipy.io as sio
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02 #for use with control.Environment(... control_timestep=#)
# _CONTROL_SUBSTEPS = 100 #for use with control.Environment(... n_sub_steps=#)

SUITE = containers.TaggedTasks()

width = 500 #for rendering
height = 500

vec = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]])#*-1
goalAngles = np.array([[180, 180], [150, 90], [90, 180], [60, 60]])
handle = collections.namedtuple('handle', ['goalQ', 'alphaNorm', 'qpos', 'oldDeltaQ', 'physics', 'out', 'vec', 'qvel', 'qacc', 'outMax'])    

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_time', 90, 'Maximum time for plotting/playback')
flags.DEFINE_integer('fpsOut', 30, 'Frame step for plotting/playback')
flags.DEFINE_integer('dpi', 300, 'DPI for mp4')
# flags.DEFINE_integer('goalAngles', "10, 45", 'Joint angles to accomplish')
flags.DEFINE_float('anglePause', 3, 'Time to pause at each joint angle')
flags.DEFINE_string('model_filename', 'ratMocap\\models\\localMotor.xml', 'filename for model.')

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(FLAGS.model_filename), common.ASSETS

class localMotor(base.Task):
    """A task for the localMotor arm."""

    def __init__(self, random=None):
        """Initializes an instance of `localMotor`.

        Args:
            random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super(localMotor, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets a random collision-free configuration at the start of each episode.

        Args:
            physics: An instance of `Physics`.
        """
        penetrating = True
        while penetrating:
            randomizers.randomize_limited_and_rotational_joints(
                physics, self.random)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0

    def get_observation(self, physics):
        """Returns a set of egocentric features."""
        obs = collections.OrderedDict()
        obs['joint_angles'] = physics.joint_angles()
        return obs

    def move(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
        """Returns the move task."""
        physics = mujoco.Physics.from_xml_string(*get_model_and_assets())
        task = localMotor(random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        #################################NEEDS TO BE WRITTEN

        return 0

def getFrame(h):
    oldDeltaQ = h.oldDeltaQ
    out = h.out
    timer3 = h.physics.time()
    while (h.physics.time() - timer3) < 1/FLAGS.fpsOut:        
        
        h.physics.set_control(out[:])
        h.physics.step()

        qpos = 180 - np.degrees(h.physics.data.qpos[:2]) #joints are being measured in wrong direction =/        
        newDeltaQ = qpos[:2] - h.goalQ####=comparator (edit index to select joints)
        
        dOut = np.sum(h.vec*newDeltaQ, 1)
        dOut = dOut*(dOut > 0)#comparator out signals to integrators
        
        # anti = h.vec*newDeltaQ/h.alphaNorm
        # anti = np.sum(anti*(dOut > 0), 0)

        # anti = out*(dOut > 0)

        # anti = np.sum((dOut > 0)*(h.vec*newDeltaQ/h.alphaNorm).transpose(), 1)
        # anti = np.sum(anti*(h.vec > 0), 1)
        anti = vec*np.sum(((dOut >= 0)*out*vec.transpose()), 1)#/h.alphaNorm
        anti = np.sum(anti*(anti < 0),1)#cross inhibition from opposing muscle groups, gated by comparator output

        # dDeltaQ = newDeltaQ - oldDeltaQ
        # alpha = np.sum(h.vec*dDeltaQ/h.alphaNorm, 1)##################should be different than this....    
        noise = np.random.wald(0.0001, 0.001, out.shape)*(1*(out == 0) - 1*(out > 0))#np.random.wald(alpha/1000 + 0.00001*(alpha <= 0), 0.001, out.shape)*(1*(dOut == 0) - 1*(dOut > 0))   
        # alpha = 1 + alpha*(alpha < 0) - anti/h.outMax - noise
        alpha = 1 + anti/h.outMax + noise#(noise positive if no increase in output, negative otherwise)(anti is already negative)
        alpha = alpha*(alpha > 0)
     
        out = dOut + alpha*out
        out = out*(out <= h.outMax)*(0 < out) + h.outMax*(out > h.outMax)

        oldDeltaQ = newDeltaQ
    vidFrame = np.hstack([h.physics.render(height, width, camera_id="fixed")])
    h = handle(h.goalQ, h.alphaNorm, qpos, oldDeltaQ, h.physics, out, h.vec, h.physics.data.qvel, h.physics.data.qacc, h.outMax)
    return vidFrame, h

def main(unused_argv):
    env = localMotor.move()
    physics = env.physics
    max_frame = FLAGS.max_time*FLAGS.fpsOut + 1
    video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)
    i1 = 0
    oldDeltaQ = np.zeros(goalAngles.shape[1])
    timer1 = physics.time()
    
    out = np.zeros(vec.shape[0])
    outMax = np.sum(abs(physics.model.actuator_ctrlrange), 1)
    
    alpha = np.zeros(vec.shape[0])
    alphaNorm = 180 - np.degrees(np.sum(abs(physics.model.jnt_range[:2]), 1))#####edit index to select joints
    alphaNorm = alphaNorm + 1*(alphaNorm == 0)

    with progressbar.ProgressBar(max_value=max_frame, poll_interval=5) as bar:        
        while (physics.time() - timer1) < FLAGS.max_time:
            for i in goalAngles:
                pausing = False

                failing = True
                timer2 = 999999999
                
                # timer2 = physics.time()
                # failing = False

                while (physics.time() - timer1) < FLAGS.max_time and (failing or (physics.time() - timer2) < FLAGS.anglePause):
                    h = handle(i, alphaNorm, [], oldDeltaQ, physics, out, vec, [], [], outMax)
                    
                    video[i1], h = getFrame(h)
                    oldDeltaQ = h.oldDeltaQ
                    out = h.out            

                    failing = sum(abs(oldDeltaQ)) > 30*oldDeltaQ.shape[0]

                    if not failing and not pausing:
                        pausing = True
                        timer2 = physics.time()
                    elif failing and pausing:
                        pausing = False
                        timer2 = 999999999

                    bar.update(i1)
                    i1 += 1    
                # print(i)   
                # print('-------')
                print(oldDeltaQ)
                # print('*******')


    fig = plt.figure()
    plt.waitforbuttonpress()
    ticky = time.time()
    for i in range(video.shape[0]):
        if i == 0:
            img = plt.imshow(video[i])
        else:
            img.set_data(video[i])
            fig.canvas.flush_events()
        clock_dt = time.time() - ticky
        ticky = time.time()
        # Real-time playback not always possible as clock_dt > .03
        plt.draw()
        time.sleep(max(0.01, 1/FLAGS.fpsOut - clock_dt))
        # plt.pause(max(0.01, 1/fpsOut - clock_dt))  # Need min display time > 0.0.
    plt.waitforbuttonpress()

if __name__ == '__main__':
    # flags.mark_flag_as_required('fileName')
    app.run(main)