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
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers

import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_TIME_LIMIT = 20
# _CONTROL_TIMESTEP = 0.02 #for use with control.Environment(... control_timestep=#)
_CONTROL_SUBSTEPS = 100 #for use with control.Environment(... n_sub_steps=#)

SUITE = containers.TaggedTasks()

CONVERSION_LENGTH = 1000 #ASSUMES .MAT TO BE IN MILLIMETERS
width = 500 #for rendering
height = 500

FLAGS = flags.FLAGS
flags.DEFINE_string('fileName', '.\\demos\\ratMocapImputedUnregistered', 'mat file to be parsed.')
flags.DEFINE_string('varName', 'markers_preproc', 'variable name to be extracted from mat file.')
flags.DEFINE_integer('max_frame', 54000, 'Maximum number of frames for plotting/playback')
flags.DEFINE_integer('start_frame', 0, 'First frame to plot')
# flags.DEFINE_integer('fpsIn', 300, 'Frame step of recording')
flags.DEFINE_integer('fpsOut', 30, 'Frame step for plotting/playback')
flags.DEFINE_integer('dpi', 300, 'DPI for mp4')
flags.DEFINE_integer('maxRenderTime', 10, 'Maximum rendering time for plotting/playback')
flags.DEFINE_string('model_filename', 'rodent_mocap_tendon_default.xml', 'filename for model.')
flags.DEFINE_bool('qOnly', False, 'Whether to make only .mat of joint angles.')                     

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(FLAGS.model_filename), common.ASSETS

class jeffRat(base.Task):
    """A task for the Rat skeleton."""

    def __init__(self, random=None):
        """Initializes an instance of `jeffRat`.

        Args:
            random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super(jeffRat, self).__init__(random=random)

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

    def mocap(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
        """Returns the mocap task."""
        physics = mujoco.Physics.from_xml_string(*get_model_and_assets())
        task = jeffRat(random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit, n_sub_steps=_CONTROL_SUBSTEPS,
            **environment_kwargs)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        #################################NEEDS TO BE WRITTEN

        return 0

def parse(fileName, varName):
    """Parses .mat file.

    Args:
    fileName: The .mat file to be parsed.
    varName: The corresponding variable name in the mat file holding mocap data.

    Returns:
    A namedtuple with fields:
        `marks`, a dictionary array containing [x, y, z] coordinates for each frame, 
            indexed by body name.
        `bods`, is a list of dictionary keys for marks (i.e. names of indexed bodies).
        'medianPose', a dictionary array containing [x, y, z] coordinates of median pose,
            indexed by body name.
    """
    parsed = collections.namedtuple('parsed', ['marks', 'bods', 'medianPose', 'mocap_pos', 'fpsIn'])

    marks = dict()
    medianPose = dict()
    zOffset = dict()
    values = sio.loadmat(fileName, variable_names = [varName, 'fps'])
    fpsIn = values['fps'][0][0]
    bods = list(values[varName].dtype.fields)
    for bod in bods:
        marks[bod] = values[varName][bod][0][0]/CONVERSION_LENGTH #ASSUMES .MAT TO BE IN MILLIMETERS
        zOffset[bod] = np.amin(marks[bod], axis = 0) * [0, 0, 1]

    for bod in bods:
        marks[bod] -= zOffset[min(zOffset)]

    for bod in bods:
        if bod == bods[0]:
            mocap_pos = marks[bod]
        else:
            mocap_pos = np.concatenate((mocap_pos, marks[bod]), axis = 1)
        medianPose[bod] = np.median(marks[bod], axis = {0})

    mocap_pos.shape = (mocap_pos.shape[0], int(mocap_pos.shape[1]/3), 3)
    mocap_pos = forFillNan(mocap_pos)

    return parsed(marks, bods, medianPose, mocap_pos, fpsIn)

def forFillNan(arr):
    print('Filling NaNs', end='')
    for m in range(arr.shape[1]):
        if any(np.isnan(arr[:, m, 1])):
            ind = np.where(np.isnan(arr[:, m, 1])) #only works if all coordinates are NaNs    
            for i in ind[0]:
                if i == 0:
                    arr[i, m, :] = [0, 0, 0]
                else:
                    arr[i, m, :] = arr[i - 1, m, :].copy()
                print('>', end='')
        print('.', end='')
    print('done.')
    return arr

def getFrame(data, env, i):
    p_i = data.mocap_pos[i, :].copy()
    with env.physics.reset_context():
        env.physics.data.mocap_pos[:] = p_i
    while (env.physics.time() < 0.3 or any(abs(env.physics.data.qvel) > 1e-05) or np.nanmean(abs(env.physics.data.qvel)) > 1e-06) and env.physics.time() < FLAGS.maxRenderTime:
        env.physics.step()
        # print(env.physics.time(), max(abs(env.physics.data.qvel)), np.nanmean(abs(env.physics.data.qvel)))
        if env.physics.time() <= FLAGS.maxRenderTime/2:
            print('.', end='')
        else:            
            print(':', end='')
    if env.physics.time() >= FLAGS.maxRenderTime:
        print('@@@@', end='')
    return np.hstack([env.physics.render(height, width, camera_id="front_side")]), env.physics.data.qpos[:]

def getJoints(data, env, i):
    p_i = data.mocap_pos[i, :].copy()
    with env.physics.reset_context():
        env.physics.data.mocap_pos[:] = p_i
    while (env.physics.time() < 0.3 or any(abs(env.physics.data.qvel) > 1e-05) or np.nanmean(abs(env.physics.data.qvel)) > 1e-06) and env.physics.time() < FLAGS.maxRenderTime:
        env.physics.step()
        # print(env.physics.time(), max(abs(env.physics.data.qvel)), np.nanmean(abs(env.physics.data.qvel)))
        if env.physics.time() <= FLAGS.maxRenderTime/2:
            print('.', end='')
        else:            
            print(':', end='')
    if env.physics.time() >= FLAGS.maxRenderTime:
        print('@@@@', end='')
    return env.physics.data.qpos[:]

def main(unused_argv):
    env = jeffRat.mocap()#SET SKELETON/ENVIRONMENT for MOCAP HERE
    data = parse(FLAGS.fileName, FLAGS.varName)  # Parse specified clip.

    max_frame = min(FLAGS.max_frame, data.mocap_pos.shape[0])
    frame_step = data.fpsIn//FLAGS.fpsOut
    max_num_frames = (max_frame - FLAGS.start_frame)//frame_step  
    qVid = np.zeros((max_num_frames, env.physics.data.qpos.size), dtype=np.uint8)
    if not FLAGS.qOnly:     
        # Set up formatting for the movie files
        metadata = dict(title= FLAGS.model_filename + ': ' + FLAGS.fileName, artist='Jeff Rhoades/Jesse Marshall/DeepMind',
                        comment=FLAGS.varName)
        writer = animation.FFMpegWriter(fps=FLAGS.fpsOut, metadata=metadata, bitrate=-1)
        video = np.zeros((max_num_frames, height, width, 3), dtype=np.uint8)

    #   with env.physics.reset_context():
    #       env.physics.data.qpos[:] = env.physics.model.key_qpos

    print('Getting video', end='')
    for i in range(FLAGS.start_frame, max_frame, frame_step):
        i1 = (i - FLAGS.start_frame)//frame_step 
        if not FLAGS.qOnly:
            video[i1], qVid[i1] = getFrame(data, env, i)
        else:
            qVid[i1] = getJoints(data, env, i)
        print('>')
    print('done.')

    out = dict()
    out['qpos'] = qVid
    out['fpsIn'] = data.fpsIn
    out['fpsOut'] = FLAGS.fpsOut
    today = datetime.datetime.today()
    if not FLAGS.qOnly:      
        out['vid'] = video
        outName = FLAGS.fileName + "_vid" + today.strftime('%Y%m%d_%H%M')
    else:
        outName = FLAGS.fileName + "_qvid" + today.strftime('%Y%m%d_%H%M')

    print('Saving Matlab file................................................', end='')
    sio.savemat(outName, out, do_compression=True)
    print('done.')

    if not FLAGS.qOnly:
        print('Saving mp4', end='')
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

if __name__ == '__main__':
    # flags.mark_flag_as_required('fileName')
    app.run(main)