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

    python mocap_demo --filename='path/to/mocap.mat'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Internal dependencies.

from absl import app
from absl import flags

from dm_control.suite import jeffRat
from dm_control.suite.utils import parse_mat_rat

import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'mat file to be parsed.')
flags.DEFINE_string('varName', None, 'variable name to be extracted from mat file.')
flags.DEFINE_integer('max_num_frames', 54000,
                     'Maximum number of frames for plotting/playback')


def main(unused_argv):
  env = jeffRat.mocap()#SET SKELETON/ENVIRONMENT for MOCAP HERE
  # Parse specified clip.
  data = parse_mat_rat.parse(FLAGS.filename, FLAGS.varName)

  max_frame = min(FLAGS.max_num_frames, data.mocap_pos.shape[0])

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  for i in range(max_frame):
    p_i = data.mocap_pos[i, :]
    with env.physics.reset_context():
      env.physics.data.mocap_pos[:] = p_i
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

  tic = time.time()
  for i in range(max_frame):
    if i == 0:
      img = plt.imshow(video[i])
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
  flags.mark_flag_as_required('filename')
  flags.mark_flag_as_required('varName')
  app.run(main)
