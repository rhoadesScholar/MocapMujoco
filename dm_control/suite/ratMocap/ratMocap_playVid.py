#ratMocap_playVid.py

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

    python ratMocap --fileName='path/to/mocap.mat'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from pathlib import Path
import os
import progressbar
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
# _CONTROL_TIMESTEP = 0.02 #for use with control.Environment(... control_timestep=#)
_CONTROL_SUBSTEPS = 100 #for use with control.Environment(... n_sub_steps=#)

SUITE = containers.TaggedTasks()

CONVERSION_LENGTH = 1000 #ASSUMES .MAT TO BE IN MILLIMETERS
width = 500 #for rendering
height = 500

FLAGS = flags.FLAGS
flags.DEFINE_string('fileName', 'dataOutput\\ratMocapOutExample.mat', 'mat file to be parsed.')
flags.DEFINE_integer('max_frame', 9999999999, 'Maximum number of frames for plotting/playback')
flags.DEFINE_integer('start_frame', 0, 'First frame to plot')
# flags.DEFINE_integer('fpsIn', 300, 'Frame step of recording')
flags.DEFINE_integer('fpsOut', 30, 'Frame step for plotting/playback')
flags.DEFINE_integer('dpi', 300, 'DPI for mp4')
flags.DEFINE_string('model_filename', 'models\\ratMocap.xml', 'filename for model.')
flags.DEFINE_bool('record', False, 'Whether to write a video file.')     
flags.DEFINE_bool('mocap', True, 'Whether render mocap markers.')                           


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

def parse(fileName):
    """Parses .mat file already rendered from mocap

    Args:
    fileName: The .mat file to be parsed.

    Returns:
    A namedtuple with fields:
        'qVid' - containing joint angles
        'fpsIn'
        'badFrame' - boolean array indicating frames that rendered for the maximum allowed time
    """
    parsed = collections.namedtuple('parsed', ['qVid', 'badFrame', 'fpsIn', 'mocap_pos'])

    values = sio.loadmat(fileName, variable_names = ['qpos', 'badFrame', 'fpsOut', 'mocap_pos'])
    fpsIn = values['fpsOut'][0][0]
    badFrame = values['badFrame'][0]    
    qVid = values['qpos']   
    mocap_pos = values['mocap_pos']

    return parsed(qVid, badFrame, fpsIn, mocap_pos)

def main(unused_argv):
    
    print('Setting up...')
    env = jeffRat.mocap()#SET SKELETON/ENVIRONMENT for MOCAP HERE
    data = parse(FLAGS.fileName)  # Parse specified clip.

    max_frame = min(FLAGS.max_frame, data.qVid.shape[0])
    fpsOut = min(data.fpsIn, FLAGS.fpsOut)
    frame_step = data.fpsIn//fpsOut
    max_num_frames = (max_frame - FLAGS.start_frame)//frame_step + 1
    video = np.zeros((max_num_frames, height, width, 3), dtype=np.uint8)
    
    print('Getting video...')
    with progressbar.ProgressBar(max_value=max_num_frames, poll_interval=3) as bar:
        for i in range(FLAGS.start_frame, max_frame, frame_step):        
            i1 = (i - FLAGS.start_frame)//frame_step
            p_i = data.qVid[i, :]
            with env.physics.reset_context():
                env.physics.data.qpos[:] = p_i
                if FLAGS.mocap:
                    env.physics.data.mocap_pos[:] = data.mocap_pos[i].copy()
                env.physics.step()
                video[i1] = np.hstack([env.physics.render(height, width, camera_id='side')])
            bar.update(i1)

    fig = plt.figure()
    plt.waitforbuttonpress()
    tic = time.time()
    for i in range(video.shape[0]):
        if i == 0:
            img = plt.imshow(video[i])
        else:
            img.set_data(video[i])
            fig.canvas.flush_events()
        toc = time.time()
        clock_dt = toc - tic
        tic = time.time()
        # Real-time playback not always possible as clock_dt > .03
        plt.draw()
        time.sleep(max(0.001, 1/fpsOut - clock_dt))
        # plt.pause(max(0.01, 1/fpsOut - clock_dt))  # Need min display time > 0.0.
    plt.waitforbuttonpress()
    
    if FLAGS.record:
        print('Saving mp4', end='')

        vidFile = Path(FLAGS.fileName)
        modelFile = Path(FLAGS.model_filename)
        path = os.fspath(vidFile.parent)
        outName = os.fspath(Path(path, vidFile.stem + "_via_" + modelFile.stem + "_" + str(FLAGS.start_frame) + "_thru_" + str(max_frame)))
        
        metadata = dict(title= FLAGS.model_filename + ': ' + FLAGS.fileName, artist='Jeff Rhoades/Jesse Marshall/DeepMind',
                        comment=FLAGS.varName)
        writer = animation.FFMpegWriter(fps=fpsOut, metadata=metadata, bitrate=-1)
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