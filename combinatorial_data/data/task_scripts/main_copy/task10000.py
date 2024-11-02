import random
import numpy as np
import phyre.creator as creator_lib
from phyre.component_utils import *

object_list = [
    'dynamic_ball_1',
    'dynamic_ball_2',
    'static_ball', 
    'dynamic_bar',
    'static_bar', # 0/1/2 obstacle(s)
    'dynamic_jar',
    'static_jar',
    'dynamic_standingsticks',
    'static_standingsticks',
]

@creator_lib.define_task_template(
    object1_x=np.linspace(0.05, 0.95, 19),
    object1_r=np.linspace(0.06, 0.12, 3),
    object1_y=np.linspace(0.2, 0.8, 5),
    object1_angle=np.linspace(-180, 180, 13),
    object2_x=np.linspace(0.05, 0.95, 19),
    object2_r=np.linspace(0.06, 0.12, 3),
    object2_y=np.linspace(0.2, 0.8, 5),
    object2_angle=np.linspace(-180, 180, 13),
    object3_x=np.linspace(0.05, 0.95, 19),
    object3_r=np.linspace(0.06, 0.12, 3),
    object3_y=np.linspace(0.2, 0.8, 5),
    object3_angle=np.linspace(-180, 180, 13),
    object4_x=np.linspace(0.05, 0.95, 19),
    object4_r=np.linspace(0.06, 0.12, 3),
    object4_y=np.linspace(0.2, 0.8, 5),
    object4_angle=np.linspace(-180, 180, 13),
    version='2',
)
def build_task(C, 
    object1_x, object1_r, object1_y, object1_angle,
    object2_x, object2_r, object2_y, object2_angle,
    object3_x, object3_r, object3_y, object3_angle,
    object4_x, object4_r, object4_y, object4_angle,
    ):

    K = 4

    # random sample k elements from the list
    objects = random.sample(object_list, K)
    instance_list = []

    for i, obj in enumerate(objects):
        if i == 0:
            x, r, y, angle = object1_x, object1_r, object1_y, object1_angle
        elif i == 1:
            x, r, y, angle = object2_x, object2_r, object2_y, object2_angle
        elif i == 2:
            x, r, y, angle = object3_x, object3_r, object3_y, object3_angle
        elif i == 3:
            x, r, y, angle = object4_x, object4_r, object4_y, object4_angle
        
        if obj == 'dynamic_ball_1':
            instance = build_dynamic_ball(C, x, r, y, 0)
        elif obj == 'dynamic_ball_2':
            instance = build_dynamic_ball(C, x, r, y, 1)
        elif obj == 'static_ball':
            instance = build_static_ball(C, x, r, y)
        elif obj == 'dynamic_bar':
            instance = build_dynamic_bar(C, x, y, r, angle)
        elif obj == 'static_bar':
            instance = build_static_bar(C, x, y, r, angle)
        elif obj == 'dynamic_jar':
            instance = build_dynamic_jar(C, x, y, r)
        elif obj == 'static_jar':
            instance = build_static_jar(C, x, y, r)
        elif obj == 'dynamic_standingsticks':
            instance = build_dynamic_standingsticks(C, x, y, r, angle)
        elif obj == 'static_standingsticks':
            instance = build_static_standingsticks(C, x, y, r, angle)
        else:
            raise ValueError(f'Unknown object: {obj}')

        instance_list.append(instance)

    # select two objects from the list: The first is dynamic, the second is anyone execept the first
    body1, body2 = random.sample(instance, 2)

    # Create assignment:
    C.update_task(
        body1=body1,
        body2=body2,
        relationships=[C.SpatialRelationship.TOUCHING])
    C.set_meta(C.SolutionTier.BALL)