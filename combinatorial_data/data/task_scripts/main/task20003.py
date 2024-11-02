# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Template task in which two balls should touch each other."""
import numpy as np
import phyre.creator as creator_lib


@creator_lib.define_task_template(
    seed=range(100),
    max_tasks=100,
)
def build_task(C, seed):

    bottom_wall = C.add('static bar', 1, bottom=0, left=0)
    ball1 = C.add(
        'dynamic ball',
        scale=0.1,
        bottom=bottom_wall.top, 
        left=0.02 * C.scene.width)
     

    block = C.add('static ball', scale=0.04,                 
                    center_x=C.scene.width * (0.4 + (seed - 50) * 0.0001),
                    center_y=C.scene.height * 0.3)
    

    # Create assignment:
    C.update_task(
        body1=ball1,
        body2=bottom_wall,
        relationships=[C.SpatialRelationship.TOUCHING])
    C.set_meta(C.SolutionTier.BALL)
