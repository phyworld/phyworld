import numpy as np
import random

def build_dynamic_ball(C, ball_id):
    ball_r=np.random.uniform(0.06, 0.12)
    ball_x=np.random.uniform(0.3, 0.6)
    height=np.random.uniform(0.4, 0.85)

    assert ball_id in [0, 1]
    ball = C.add(
        'dynamic ball',
        scale=ball_r,
        center_x=ball_x * C.scene.width,
        bottom=height * C.scene.height)
    return ball

def build_static_ball(C, seed):

    def gen_chain(start_x, start_y):
        angle = rng.uniform() * 2 * np.pi
        angle_diff = rng.uniform() * 2 * np.pi / 10
        stars = [(start_x, start_y)]
        line_length = 1
        n_valid = 0
        top = 0.5
        while n_valid < max_poitns:
            if line_length >= 3 and rng.uniform() < 0.2:
                x, y = stars[rng.choice(len(stars))]
                line_length = 1
                angle = rng.uniform() * 2 * np.pi
                angle_diff = rng.uniform() * 2 * np.pi / 10
            else:
                line_length += 1
                step = rng.uniform(0.05, 0.2)
                angle += angle_diff
                dx, dy = step * np.cos(angle), step * np.sin(angle)
                x, y = stars[-1]
                x += dx
                y += dy
            if y >= top:
                continue
            stars.append((x, y))
            if 0.0 < x < 1 and 0.0 < y < 1:
                n_valid += 1
        return stars

    rng = np.random.RandomState(seed=seed)
    max_poitns = random.randint(2, 6)

    stars = gen_chain(0.4, 0.55)

    for i, (x, y) in enumerate(stars):
        if 0 <= x <= 1 and 0 <= y <= 1:
            ball = C.add(
                'static ball',
                scale=0.04,
                center_x=C.scene.width * x,
                center_y=C.scene.height * y)

    return ball

def build_dynamic_bar(C):
    scale=np.random.uniform(0.15, 0.5)
    x=np.random.uniform(0.3, 0.6)
    y=np.random.uniform(0.2, 0.8)
    angle=np.random.uniform(-90, 90)

    obj = C.add('dynamic bar', scale) \
        .set_center_x(x * C.scene.width) \
        .set_bottom(y * C.scene.height) \
        .set_angle(angle)
    return obj

def build_dynamic_standing_bars(C):
    offset=np.random.uniform(0.45, 0.6)
    # Set parameters of bars.
    multiplier = 0.15

    # Add bars with increasing height.
    if np.random.uniform() < 0.5:
        multiplier = -multiplier
        offset = 1.0 - offset

    bars = []
    num_bars = random.randint(1, 4)
    for i, idx in enumerate(range(-num_bars//2, -num_bars//2 + num_bars)):
        bar_scale = 0.2 + 0.05 * i
        bars.append(
            C.add(
                'dynamic bar',
                scale=bar_scale,
                angle=90,
                bottom=0,
                left=(offset + multiplier * idx) * C.scene.width))

    return bars[-1]   

def build_static_bar(C):
    scale=np.random.uniform(0.2, 0.5)
    y=np.random.uniform(0.1, 0.5)
    angle=np.random.uniform(-60, 60)

    # if np.random.uniform() < 0.5:
    x=np.random.uniform(0.3, 0.4)
    obj = C.add('static bar', scale) \
        .set_right(x * C.scene.width) \
        .set_bottom(y * C.scene.height) \
        .set_angle(angle)
    # else:
    #     x=np.random.uniform(0.8, 0.95)
    #     obj = C.add('static bar', scale) \
    #         .set_right(x * C.scene.width) \
    #         .set_bottom(y * C.scene.height) \
    #         .set_angle(angle)

    return obj

def build_dynamic_jar(C):
    scale=np.random.uniform(0.08, 0.3)
    x=np.random.uniform(0.3, 0.6)
    y=np.random.uniform(0.2, 0.8)
    angle=np.random.uniform(-45, 45)

    if random.random() < 0.7:
        obj = C.add('dynamic jar', 
        scale=scale,
        bottom=y * C.scene.height,
        center_x=x * C.scene.width,
        angle=angle,
        )
    else:
        obj = C.add('dynamic jar', 
            scale=scale,
            bottom=0,
            center_x=x * C.scene.width,
            angle=0,
        )
    return obj

def build_static_jar(C):
    scale=np.random.uniform(0.08, 0.3)
    x=np.random.uniform(0.3, 0.6)
    y=np.random.uniform(0.2, 0.5)
    angle=np.random.uniform(-90, 90)

    obj = C.add('static jar', 
        scale=scale,
        bottom=y * C.scene.height,
        center_x=x * C.scene.width,
        angle=angle)
    return obj

def build_static_standingsticks(C):
    obj = C.add('static standingsticks', 
        scale=stick_scale,
        bottom=stick_y * C.scene.height,
        center_x=stick_x * C.scene.width,
        angle=stick_angle)
    return obj

def build_dynamic_standingsticks(C):
    scale=np.random.uniform(0.3, 0.6)
    x=np.random.uniform(0.3, 0.6)
    y=np.random.uniform(0.2, 0.5)
    angle=np.random.uniform(-90, 90)

    if random.random() < 0.3:
        obj = C.add('dynamic standingsticks', 
            scale=scale,
            bottom=0,
            center_x=x * C.scene.width,
            )
    else:
        obj = C.add('dynamic standingsticks', 
            scale=scale,
            bottom=y * C.scene.height,
            center_x=x * C.scene.width,
            angle=angle)
    return obj