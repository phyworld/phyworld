Here's a revised version of your README to improve 

# Data Generation Code for PAPER [How Far is Video Generation from World Model: A Physical Law Perspective]()


## In-Distribution and Out-of-Distribution Data

DIrectory `id_ood_data` contains code for generating training and evaluation data to test scaling abilities for in-distribution (in-dist) and out-of-distribution (ood) scenarios. It supports generating videos for scenarios including uniform motion, collision, and parabolic motion.


### Training Data

To generate collision videos at different data size levels:
```bash
# Collision videos with increasing data sizes
python3 two_balls_collision.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64
python3 two_balls_collision.py --data_name in_dist_v2 --data_size_level 2 --num_workers 64
python3 two_balls_collision.py --data_name in_dist_v2 --data_size_level 3 --num_workers 64
```

To generate videos of uniform motion at different data size levels (e.g., 30k, 300k, 3M videos):
```bash
python3 one_ball_uniform_motion.py --data_name in_dist_v2 --data_size_level 0 --num_workers 64
python3 one_ball_uniform_motion.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64
python3 one_ball_uniform_motion.py --data_name in_dist_v2 --data_size_level 2 --num_workers 64
```

To generate parabolic motion videos:
```bash
python3 one_ball_parabola.py --data_name in_dist_v2 --data_size_level 0 --num_workers 64
python3 one_ball_parabola.py --data_name in_dist_v2 --data_size_level 1 --num_workers 64
python3 one_ball_parabola.py --data_name in_dist_v2 --data_size_level 2 --num_workers 64
```

**Note:** The `num_workers` parameter specifies the number of parallel threads used for data generation. Adjust this based on your available CPU resources.

### Evaluation Data (In-Distribution and Out-of-Distribution)

To generate evaluation data for visualization across different scenarios:
```bash
# Collision videos for evaluation
python3 two_balls_collision.py --data_for_vis

# Uniform motion videos for evaluation
python3 one_ball_uniform_motion.py --data_for_vis

# Parabolic motion videos for evaluation
python3 one_ball_parabola.py --data_for_vis
```

---

## Combinatorial Data

We build combinatorial data generation on the [Phyre](https://github.com/facebookresearch/phyre/tree/main) codebase. Follow the installation instructions in the Phyre repository to set up the `combinatorial_data` directory.

### Training Data Generation from 60 Templates

Run the following command to generate training data from 60 templates:
```bash
# Replace $ID with values 0, 1, 2, 3, 4 and 5, with each ID generating 10 templates, totally 60 templates
python3 data_generator_v2.py --num_workers 64 --run_id $ID --data_dir ./train
```

### Template Subsets for Training

For scaling analysis, you can use a subset of the training data:
- **6 templates**: 10003, 10005, 10016, 10023, 10024, 10053
- **30 templates**: Use the regular expression `100[0-5][02468]` to select templates.

### Evaluation Data from Reserved Templates

To generate evaluation data from 10 reserved templates:
```bash
python3 data_generator_v2.py --num_workers 64 --run_id 6 --data_dir ./eval
```

## Notes

The code has been reorganized, which may lead to errors or deviations from the original research results. If you encounter any issues, please report them by opening an issue. We will address any bugs promptly.

---

## Citation

```
```
