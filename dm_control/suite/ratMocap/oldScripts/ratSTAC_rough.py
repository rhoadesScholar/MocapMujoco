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
#     py ratSTAC_rough.py --fileName='ratmocap.mat' --varName='markers_aligned_preproc'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Internal dependencies.

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

import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_TIME_LIMIT = 20
# _CONTROL_TIMESTEP = 0.02 #for use with control.Environment(... control_timestep=#)
_CONTROL_SUBSTEPS = 100 #for use with control.Environment(... n_sub_steps=#)

SUITE = containers.TaggedTasks()

CONVERSION_LENGTH = 1000 #ASSUMES .MAT TO BE IN MILLIMETERS

FLAGS = flags.FLAGS
flags.DEFINE_string('fileName', '.\\demos\\ratmocap.mat', 'mat file to be parsed.')
flags.DEFINE_string('varName', 'markers_aligned_preproc', 'variable name to be extracted from mat file.')
flags.DEFINE_integer('max_num_frames', 1250,
                     'Maximum number of frames for plotting/playback')
flags.DEFINE_integer('start_frame', 0,
                     'First frame to plot')
flags.DEFINE_integer('frame_step', 2,
                     'Frame step for plotting/playback')
flags.DEFINE_string('model_filename', 
                    r'C:\Users\RatControl\Documents\GitHub\dm_control\dm_control\suite\rodent_mocap_tendon.xml',
                    'filename for model.')                     

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
  parsed = collections.namedtuple('parsed', ['marks', 'bods', 'medianPose', 'mocap_pos', 'medianMat'])
  
  marks = dict()
  medianPose = dict()
  zOffset = dict()
  values = sio.loadmat(fileName, variable_names = varName)
  bods = list(values[varName].dtype.fields)
  for bod in bods:
    marks[bod] = values[varName][bod][0][0]/CONVERSION_LENGTH #ASSUMES .MAT TO BE IN MILLIMETERS
    zOffset[bod] = np.amin(marks[bod], axis = 0) * [0, 0, 1]
  
  if min(zOffset[min(zOffset)]) < 0:
    for bod in bods:
      marks[bod] += abs(zOffset[min(zOffset)])
  
  for bod in bods:
    if bod == bods[0]:
      mocap_pos = marks[bod]
    else:
      mocap_pos = np.concatenate((mocap_pos, marks[bod]), axis = 1)
    medianPose[bod] = np.median(marks[bod], axis = {0})
  
  for bod in bods:
    if bod == bods[0]:
      medianMat = medianPose[bod]
    else:
      medianMat = np.concatenate((medianMat, medianPose[bod]), axis = 0)

  mocap_pos.shape = (mocap_pos.shape[0], int(mocap_pos.shape[1]/3), 3)
  medianMat.shape = (int(medianMat.shape[0]/3), 3)

  return parsed(marks, bods, medianPose, mocap_pos, medianMat)

def main(unused_argv):
  
  
  env = jeffRat.mocap()#SET SKELETON/ENVIRONMENT for MOCAP HERE
  # Parse specified clip.
  data = parse(FLAGS.fileName, FLAGS.varName)

  max_frame = min(FLAGS.max_num_frames, data.mocap_pos.shape[0])

  width = 500
  height = 500
  video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)

  with env.physics.reset_context():
      env.physics.data.qpos[:] = env.physics.model.key_qpos
      env.physics.data.mocap_pos[:] = data.medianMat
  
#   SET INITIAL PARAMETERS: env.model.site_pos based data.txt file or env.data.mocap_pos
  
  
  
  for i in range(FLAGS.start_frame, max_frame, FLAGS.frame_step):
    p_i = data.mocap_pos[i, :] 

    with env.physics.reset_context():
    #   env.physics.data.qpos[:] = 
      env.physics.data.mocap_pos[:] = p_i
    #   env.physics.data.mocap_pos[:] = data.medianMat

    video[i] = np.hstack([env.physics.render(height, width, camera_id="front_side")])

  tic = time.time()
  for i in range(FLAGS.start_frame, max_frame, FLAGS.frame_step):
    if i == 0:
      img = plt.imshow(video[i])
      plt.waitforbuttonpress()
    else:
      img.set_data(video[i])
    toc = time.time()
    clock_dt = toc - tic
    tic = time.time()
    # Real-time playback not always possible as clock_dt > .03
    plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
    plt.draw()
  plt.waitforbuttonpress()


if __name__ == '__main__':
    # flags.mark_flag_as_required('fileName')
    # flags.mark_flag_as_required('varName')
    app.run(main)