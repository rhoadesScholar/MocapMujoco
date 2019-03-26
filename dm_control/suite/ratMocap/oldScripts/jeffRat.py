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

"""jeffRat Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 20
# _CONTROL_TIMESTEP = 0.02 #for use with control.Environment(... control_timestep=#)
_CONTROL_SUBSTEPS = 100 #for use with control.Environment(... n_sub_steps=#)

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('rodent_mocap_tendon.xml'), common.ASSETS

# @SUITE.add()
# def mocap(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the mocap task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = jeffRat(random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
#       **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the rat domain."""

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.  


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

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    #################################NEEDS TO BE WRITTEN
    
    return 0
    
  def mocap(self, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
      """Returns the mocap task."""
      physics = Physics.from_xml_string(*get_model_and_assets())
      task = jeffRat(random=random)
      environment_kwargs = environment_kwargs or {}
      return control.Environment(
          physics, task, time_limit=time_limit, n_sub_steps=_CONTROL_SUBSTEPS,
          **environment_kwargs)
  
