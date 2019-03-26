#localMotor_Vid.py

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

"""Demonstration of mat parsing for rat mocap.

To run the demo, supply a path to a `.mat` file:

    python mocap_demo --fileName='path/to/mocap.mat'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime

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

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_time', 90, 'Maximum time for plotting/playback')
flags.DEFINE_integer('fpsOut', 30, 'Frame step for plotting/playback')
flags.DEFINE_integer('dpi', 300, 'DPI for mp4')
flags.DEFINE_integer('goalAngles', "10, 45", 'Joint angles to accomplish')
flags.DEFINE_float('anglePause', 10, 'Time to pause at each joint angle')

# flags.DEFINE_float('qvelMax', 1e-05, 'Joint velocity threshold to continue rendering')
# flags.DEFINE_float('qvelMean', 1e-06, 'Joint velocity mean threshold to continue rendering')
# flags.DEFINE_float('qaccMax', 1e-05, 'Joint acceleration threshold to continue rendering')
# flags.DEFINE_float('qaccMean', 1e-06, 'Joint acceleration mean threshold to continue rendering')

flags.DEFINE_string('model_filename', 'ratMocap\\models\\localMotor.xml', 'filename for model.')
flags.DEFINE_bool('qOnly', False, 'Whether to make only .mat of joint angles.')
flags.DEFINE_bool('record', False, 'Whether to write a video file.')
flags.DEFINE_bool('play', True, 'Whether to play a video.')       
flags.DEFINE_bool('save', False, 'Whether to save the rendering.')       

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
            physics, task, time_limit=time_limit, n_sub_steps=_CONTROL_SUBSTEPS,
            **environment_kwargs)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        #################################NEEDS TO BE WRITTEN

        return 0

def getFrame(angle, physics):
    with physics.reset_context():
        for bod, mark in data.marks.items(): physics.named.data.mocap_pos[bod] = mark[i].copy()
    while (physics.time() < 1/FLAGS.fpsOut) :
        physics.step()
        print('.', end='')
    vidFrame = np.hstack([physics.render(height, width, camera_id="front_side")])
    return vidFrame, physics.data.qpos[:], physics.data.qvel[:], physics.data.qacc[:], physics.data.xpos[:], physics.data.actuator_length[:], physics.data.actuator_velocity[:]

def getJoints(angle, physics):
    # p_i = data.mocap_pos[i, :].copy()
    badFrame = False
    with physics.reset_context():
        # physics.data.mocap_pos[:] = p_i
        for bod, mark in data.marks.items(): physics.named.data.mocap_pos[bod] = mark[i].copy()
    while (physics.time() < FLAGS.minRenderTime or any(abs(physics.data.qvel) > FLAGS.qvelMax) or np.nanmean(abs(physics.data.qvel)) > FLAGS.qvelMean
        or any(abs(physics.data.qacc) > FLAGS.qaccMax) or np.nanmean(abs(physics.data.qacc)) > FLAGS.qaccMean) and physics.time() < FLAGS.maxRenderTime:
        physics.step()
        # print(physics.time(), max(abs(physics.data.qvel)), np.nanmean(abs(physics.data.qvel)))
        if physics.time() <= FLAGS.maxRenderTime/2:
            print('.', end='')
        else:            
            print(':', end='')
    if physics.time() >= FLAGS.maxRenderTime:
        print('@@@@', end='')
        badFrame = True
    return physics.data.qpos[:], physics.data.ten_length[:], physics.data.ten_velocity[:], badFrame, physics.data.qvel[:], physics.data.qacc[:], physics.data.xpos[:]

def main(unused_argv):
    env = localMotor.move()
    phsyics = env.physics
    max_frame = FLAGS.max_time*FLAGS.fpsOut
    qpos = np.zeros((max_frame, physics.data.qpos.size), dtype=np.float64)
    qvel = np.zeros((max_frame, physics.data.qvel.size), dtype=np.float64)
    qacc = np.zeros((max_frame, physics.data.qacc.size), dtype=np.float64)

    xpos = np.zeros((max_frame, physics.data.xpos.shape[0], physics.data.xpos.shape[1]), dtype=np.float64)
    
    muscleLen = np.zeros((max_frame, physics.data.actuator_length.size), dtype=np.float64)
    muscleVel = np.zeros((max_frame, physics.data.actuator_velocity.size), dtype=np.float64)

    if FLAGS.play or FLAGS.record or not FLAGS.qOnly:     
        # Set up formatting for the movie files
        video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)
        if FLAGS.record:
            metadata = dict(title= FLAGS.model_filename + ': ' + FLAGS.fileName, artist='Jeff Rhoades/Jesse Marshall/DeepMind',
                            comment=FLAGS.varName)
            writer = animation.FFMpegWriter(fps=FLAGS.fpsOut, metadata=metadata, bitrate=-1)

    print('Getting video', end='')
    
    with progressbar.ProgressBar(max_value=max_frame, poll_interval=5) as bar:        
        while physics.time()
            i1 = (i - FLAGS.start_frame)//frame_step
            if FLAGS.play or FLAGS.record or not FLAGS.qOnly:
                video[i1], qpos[i1], qvel[i1], qacc[i1], xpos[i1], muscleLen[i1], muscleVel[i1] = getFrame(angle, physics)
            else:
                qpos[i1], qvel[i1], qacc[i1], xpos[i1], muscleLen[i1], muscleVel[i1] = getJoints(angle, physics)
            bar.update(i1)
    print('done.')

    if FLAGS.save:
        qnames = []
        qnames[:5] = [physics.named.data.qpos.axes[0].names[0]]*6 #First 7 are pos and quaternion of root frame
        qnames.extend(physics.named.data.qpos.axes[0].names)

        muscleNames = physics.named.data.actuator_length.axes[0].names

        out = dict()
        out['qpos'] = qpos
        out['qvel'] = qvel
        out['qacc'] = qacc    
        out['xpos'] = xpos
        out['qnames'] = qnames
        out['muscleLen'] = muscleLen
        out['muscleVel'] = muscleVel
        out['muscleNames'] = muscleNames
        out['fpsOut'] = FLAGS.fpsOut
        out['model'] = FLAGS.model_filename
        today = datetime.datetime.today()
        modelFile = Path(FLAGS.model_filename)        
        path = modelFile.parents[1]
        if FLAGS.play or FLAGS.record or not FLAGS.qOnly:      
            out['vid'] = video
        outName = os.fspath(Path(path, "dataOutput", modelFile.stem + "_" + str(min(FLAGS.goalAngles)) + "_thru_" + str(max(FLAGS.goalAngles)) + "_fps" + str(FLAGS.fpsOut) + "_" + today.strftime('%Y%m%d_%H%M')))

        print('Saving Matlab file to' + outName + '................................................', end='')
        try:
            sio.savemat(outName, out, do_compression=True)
            print('done.')
        except:
            print('failed.')

    if FLAGS.play:
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

    if FLAGS.record:
        print('Saving mp4', end='')
        try:
            fig = plt.figure()
            img = plt.imshow(video[0])
            plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\RatControl\ffmpeg\bin\ffmpeg.exe'
            with writer.saving(fig, outName + '.mp4', dpi=FLAGS.dpi):
                for i in range(video.shape[0]):
                    img.set_data(video[i])
                    plt.draw()
                    writer.grab_frame()
                    print('.', end='')
            writer.finish()
            print('done.')
        except:            
            print('.......failed.')


if __name__ == '__main__':
    # flags.mark_flag_as_required('fileName')
    app.run(main)