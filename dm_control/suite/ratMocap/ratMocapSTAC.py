#ratMocap.py

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
flags.DEFINE_string('fileName', '.\\dataInput\\ratMocapImputedUnregisteredDataIn_2', 'mat file to be parsed.')
flags.DEFINE_string('varName', 'markers_preproc', 'variable name to be extracted from mat file.')
flags.DEFINE_integer('max_frame', 99999999, 'Maximum number of frames for plotting/playback')
flags.DEFINE_integer('start_frame', 0, 'First frame to plot')
# flags.DEFINE_integer('fpsIn', 300, 'Frame step of recording')
flags.DEFINE_float('fpsOut', 60, 'Frame step for plotting/playback')
flags.DEFINE_integer('dpi', 300, 'DPI for mp4')

flags.DEFINE_string('model_filename', 'ratMocap\\models\\ratMocap.xml', 'filename for model.')
flags.DEFINE_string('outName', None, 'filename and path for output.')
flags.DEFINE_bool('qOnly', True, 'Whether to make only .mat of joint angles.')
flags.DEFINE_bool('record', False, 'Whether to write a video file.')
flags.DEFINE_bool('play', False, 'Whether to play a video.')       
flags.DEFINE_bool('save', True, 'Whether to save the rendering.')  

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

    def sensor_loss(q, physics, reference, time_index, walker_sites, reg_coef=0.):
        reg_term = reg_coef*np.sum(q[7:]**2)  # skip translation and quaternion
        residual = (reference[:, time_index] - joints_to_sensors(q, physics, walker_sites))
        return .5*np.sum(residual**2) + reg_term

    def joints_to_sensors(q, physics, walker_sites):
        physics.named.data.qpos[:] = q.copy()
        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
        mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)
        return_value = np.array(physics.bind(walker_sites).xpos)
        return return_value

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
        marks[bod] -= zOffset[min(zOffset)]#CHANGE ZOFFSET <-------------------------------------------$$$$$

    for bod in bods:
        marks[bod] = forFillNan(marks[bod])
        medianPose[bod] = np.median(marks[bod], axis = {0})

    mocap_pos = np.empty([marks[bod].shape[0], len(marks), marks[bod].shape[1]])
    for b in range(len(bods)):
        bod = bods[b]
        mocap_pos[:,b, :] = marks[bod]

    return parsed(marks, bods, medianPose, mocap_pos, fpsIn)

def forFillNan(arr): #For marks[bod]
    print('Filling NaNs> ', end='')
    if np.isnan(arr).any():
        ind = np.where(np.isnan(arr[:, 1])) #only works if all coordinates are NaNs    
        for i in ind[0]:
            if i == 0:
                arr[i, :] = [0, 0, 0]
            else:
                arr[i, :] = arr[i - 1, :].copy()
            print('.', end='')
    print('done.')
    return arr

def getFrame(data, physics, i):
    lb = np.concatenate([-np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 0]])
    lb = np.minimum(lb, 0.0)
    ub = np.concatenate([np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 1]])
	
    # q-phase
    q0 = np.copy(physics.named.data.qpos[:])
    q_opt_param = scipy.optimize.least_squares(lambda q: sensor_loss(q, physics, sensor_ref_arr, frame_index,  
	# pylint: disable=g-long-lambda walker_sites, reg_coef=reg_coef), q0, bounds=(lb, ub), ftol=_FTOL, diff_step=3e-8)
    
	physics.named.data.qpos[:] = q_opt_param.x
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    
    # m-phase
    m0 = np.copy(physics.bind(walker_sites).pos[:])
    physics.bind(walker_sites).pos[:] = ...#set via

    return np.hstack([physics.render(height, width, camera_id="side")]), physics.data.qpos[:], physics.data.xpos[:]

def getJoints(data, physics, i):
    lb = np.concatenate([-np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 0]])
	lb = np.minimum(lb, 0.0)
	ub = np.concatenate([np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 1]])
	
    # q-phase
    q0 = np.copy(physics.named.data.qpos[:])
	q_opt_param = scipy.optimize.least_squares(lambda q: sensor_loss(q, physics, sensor_ref_arr, frame_index,  
	# pylint: disable=g-long-lambda walker_sites, reg_coef=reg_coef), q0, bounds=(lb, ub), ftol=_FTOL, diff_step=3e-8)
    
	physics.named.data.qpos[:] = q_opt_param.x
	mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    
    # m-phase
    m0 = np.copy(physics.bind(walker_sites).pos[:])
    physics.bind(walker_sites).pos[:] = ...#set via

    return physics.data.qpos[:], physics.data.xpos[:]

def main(unused_argv):
    tic = time.time()
    inFile = Path(FLAGS.fileName).absolute()
    env = jeffRat.mocap()#SET SKELETON/ENVIRONMENT for MOCAP HERE
    data = parse(os.fspath(FLAGS.fileName), FLAGS.varName)  # Parse specified clip.

    max_frame = min(FLAGS.max_frame, data.mocap_pos.shape[0])
    frame_step = int(data.fpsIn//FLAGS.fpsOut)
    max_num_frames = (max_frame - FLAGS.start_frame)//frame_step + 1
    qpos = np.zeros((max_num_frames, env.physics.data.qpos.size), dtype=np.float64)
    xpos = np.zeros((max_num_frames, env.physics.data.xpos.shape[0], env.physics.data.xpos.shape[1]), dtype=np.float64)
    
    if FLAGS.play or FLAGS.record or not FLAGS.qOnly:     
        # Set up formatting for the movie files
        video = np.zeros((max_num_frames, height, width, 3), dtype=np.uint8)
        if FLAGS.record:
            metadata = dict(title= FLAGS.model_filename + ': ' + FLAGS.fileName, artist='Jeff Rhoades/Jesse Marshall/DeepMind',
                            comment=FLAGS.varName)
            writer = animation.FFMpegWriter(fps=FLAGS.fpsOut, metadata=metadata, bitrate=-1)

    for i in range(FLAGS.start_frame, max_frame, frame_step):
        i1 = (i - FLAGS.start_frame)//frame_step
        if FLAGS.play or FLAGS.record or not FLAGS.qOnly:
            video[i1], qpos[i1], xpos[i1] = getFrame(data, env.physics, i)
        else:
            qpos[i1], xpos[i1] = getJoints(data, env.physics, i)
    
    if FLAGS.save:
        qnames = []
        qnames[:5] = [env.physics.named.data.qpos.axes[0].names[0]]*6 #First 7 are pos and quaternion of root frame
        qnames.extend(env.physics.named.data.qpos.axes[0].names)

        tendonNames = env.physics.named.data.ten_length.axes[0].names

        out = dict()
        out['qpos'] = qpos
        out['xpos'] = xpos
        out['qnames'] = qnames
        out['fpsIn'] = data.fpsIn
        out['fpsOut'] = FLAGS.fpsOut
        out['mocap_pos'] = data.mocap_pos        
        out['markerNames'] = data.bods
        out['model'] = FLAGS.model_filename
        out['medianPose'] = data.medianPose
        today = datetime.datetime.today()
        path = inFile.parents[1]
        modelFile = Path(FLAGS.model_filename)
        if FLAGS.play or FLAGS.record or not FLAGS.qOnly:      
            out['vid'] = video
        
        if FLAGS.outName:
            outName = FLAGS.outName
        else:
            outName = os.fspath(Path(path, "dataOutput", inFile.stem + "_via_" + modelFile.stem + "_" + str(FLAGS.start_frame) + "_thru_" + str(max_frame) + "_fps" + str(FLAGS.fpsOut) + "_" + today.strftime('%Y%m%d_%H%M')))        

        print('Saving Matlab file to ' + outName + '................................................')
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