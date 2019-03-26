# Copyright 2019 Jeff Rhoades rhoades@g.harvard.edu.
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

"""Parse .mat motion capture data."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import collections
import scipy.io as sio

import numpy as np

CONVERSION_LENGTH = 1000 #ASSUMES .MAT TO BE IN MILLIMETERS

# _RAT_XML_JOINT_ORDER = (
#     'SpineR', 'LFemur_rx', 'LFemur_ry', 'LFemur_rz', 'LShin', 'RFemur_rx', 'RFemur_ry',
#     'RFemur_rz', 'RFemur', 'RShin', 'SpineF', 'Neck_rx', 'Neck_rz', 'Skull1', 'Skull2', 
#     'Skull3', 'LScap', 'LScap_2', 'LHumerus_rz', 'LHumerus_ry', 'LHumerus_rx', 'LArm', 
#     'RScap', 'RScap_2', 'RHumerus_rz', 'RHumerus_ry', 'RHumerus_rx', 'RArm'
# )

#py ratSTAC.py --filename='ratmocap.mat'

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
  parsed = collections.namedtuple('parsed', ['marks', 'bods', 'medianPose', 'mocap_pos'])
  
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
  
  mocap_pos.shape = (mocap_pos.shape[0], int(mocap_pos.shape[1]/3), 3)

  return parsed(marks, bods, medianPose, mocap_pos)