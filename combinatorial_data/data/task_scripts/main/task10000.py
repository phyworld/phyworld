import random
import numpy as np
import phyre.creator as creator_lib
from phyre.component_utils import *
from itertools import combinations
import os

object_list = [
    'dynamic_ball_1',
    'dynamic_ball_2',
    'static_ball', 
    'dynamic_bar',
    'dynamic_standing_bars',
    'static_bar', # 0/1/2 obstacle(s)
    'dynamic_jar',
    # 'static_jar',
    'dynamic_standingsticks',
    # 'static_standingsticks',
]

# random sample k elements from the list
K = 4
all_combinations = combinations(object_list, K)
viable_combinations = []

for objects in all_combinations:
    exit_dynamic = exit_static = False
    for obj_name in objects:
        if 'dynamic' in obj_name:
            exit_dynamic = True
        # if 'static' in obj_name:
        #     exit_static = True
    # if exit_dynamic and exit_static:
    if exit_dynamic:
        viable_combinations.append(objects)

random.Random(4).shuffle(viable_combinations)
# print(viable_combinations[:2])

# def get_index(combination):
#     return sorted([object_list.index(x) for x in combination])
# for i, c in enumerate(viable_combinations):
#     if get_index(c) in [
#         [0,1,2,3],
#         [4,5,6,7],
#         [0,1,4,5],
#         [0,1,6,7],
#         [2,3,4,5],
#         [2,3,6,7],
#     ]:
#         print(f'{10000+i:5d}', c)

# 10003 ('dynamic_standing_bars', 'static_bar', 'dynamic_jar', 'dynamic_standingsticks')
# 10005 ('static_ball', 'dynamic_bar', 'dynamic_standing_bars', 'static_bar')
# 10016 ('static_ball', 'dynamic_bar', 'dynamic_jar', 'dynamic_standingsticks')
# 10023 ('dynamic_ball_1', 'dynamic_ball_2', 'static_ball', 'dynamic_bar')
# 10024 ('dynamic_ball_1', 'dynamic_ball_2', 'dynamic_standing_bars', 'static_bar')
# 10053 ('dynamic_ball_1', 'dynamic_ball_2', 'dynamic_jar', 'dynamic_standingsticks')


@creator_lib.define_task_template(
    seed=range(1000000),
    version='2',
)
def build_task(C, 
    seed
    ):

    instance_list = []
    body1 = body2 = None
    
    template_index = int(os.path.split(__file__)[-1][5:9])
    objects = viable_combinations[template_index]

    for i, obj in enumerate(objects):
        if obj == 'dynamic_ball_1':
            instance = build_dynamic_ball(C, 0)
        elif obj == 'dynamic_ball_2':
            instance = build_dynamic_ball(C, 1)
        elif obj == 'static_ball':
            instance = build_static_ball(C, seed=seed)
        elif obj == 'dynamic_bar':
            instance = build_dynamic_bar(C)
        elif obj == 'dynamic_standing_bars':
            instance = build_dynamic_standing_bars(C)
        elif obj == 'static_bar':
            instance = build_static_bar(C)
        elif obj == 'dynamic_jar':
            instance = build_dynamic_jar(C)
        elif obj == 'static_jar':
            instance = build_static_jar(C)
        elif obj == 'dynamic_standingsticks':
            instance = build_dynamic_standingsticks(C)
        elif obj == 'static_standingsticks':
            instance = build_static_standingsticks(C)
        else:
            raise ValueError(f'Unknown object: {obj}')

        if body1 is None and 'dynamic' in obj:
            body1 = instance
        # if body2 is None and 'static' in obj:
        #     body2 = instance
        else:
            instance_list.append(instance)
    body2 = random.choice(instance_list)

    # Create assignment:
    C.update_task(
        body1=body1,
        body2=body2,
        relationships=[C.SpatialRelationship.TOUCHING],
        recolor=False,
        )
    C.set_meta(C.SolutionTier.BALL)
