<div align="center">
<h1>How Far is Video Generation from World Model: A Physical Law Perspective</h1>

[**Bingyi Kang**](https://bingykang.github.io/)<sup>\*</sup>   ·  [**Yang Yue**](https://yueyang130.github.io/)<sup>\*</sup> 
<br>
[**Rui Lu**](https://lr32768.github.io/) · [**Zhijie Lin**](https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=zh-CN) · [**Yang Zhao**](https://scholar.google.com/citations?user=uPmTOHAAAAAJ&hl=en) · [**Kaixin Wang**](https://kaixin96.github.io/) · [**Gao Huang**](https://www.gaohuang.net/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)
<br>
*Equal Contribution, in alphabetical order

<a href="https://arxiv.org/abs/2411.02385"><img src='https://img.shields.io/badge/arXiv-phyworld-red' alt='Paper PDF'></a>
<a href='https://phyworld.github.io/'><img src='https://img.shields.io/badge/Project_Page-phyworld-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/magicr/phyworld'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-phyworld-blue'></a>
</div>

We conduct a systematic study to investigate whether video generation is able to learn physical laws from videos, leveraging data and model scaling.

![Alt Text](./assets/teaser2x.gif)


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

We build combinatorial data generation on the [Phyre](https://github.com/facebookresearch/phyre/tree/main) codebase. 

For install the python env, run
```
# create conda env
conda create --yes -n phyre python=3.9
source activate phyre

# install requirements
conda install -c conda-forge sed nodejs=12 thrift-cpp=0.11.0 wget pybind11=2.6 cmake boost=1.75 setuptools pip --yes
pip install matplotlib tqdm ipywidgets yapf==0.28.0

# install our project
cd combinatorial_data
pip install -e src/python
```

We put our 70 templates 10000:10069 [here](https://github.com/phyworld/phyworld/tree/master/combinatorial_data/data/task_scripts/main) and complied bins [here](https://github.com/phyworld/phyworld/tree/master/combinatorial_data/data/generated_tasks).

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

## Evaluation

Evaluation code to parse velocity and calculate error metrics from video data see here `id_ood_data/evaluate.py`.

## TODO

- Data generation code for in-depth analysis


## Download Data

| Data Type            | Train Data (30K/300K/3M)                                                                                                                  | Eval Data                                                                                                                | Description                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Uniform Motion**   | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_eval.hdf5)                    | Eval data includes both in-distribution and out-of-distribution data                                        |
| **Parabola**         | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_eval.hdf5)                          | -          |
| **Collision**        | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_eval.hdf5)                          | -                                                                            | -                                                                                                                        | -                                                                                                           |
| **Combinatorial Data** | [In-template 6M templates00:59](https://huggingface.co/datasets/magicr/phyworld/tree/main/combinatorial_data)                                                                                                               | [Out-of-template](https://huggingface.co/datasets/magicr/phyworld/blob/main/combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5) | In-template-6M includes train data (0:990 videos in each train template) and in-template eval data (990:1000 videos in each train template). Out-template refers to eval data from reserved 10 templates (templates60:69). |

## Notes

The code has been reorganized, which may lead to errors or deviations from the original research results. If you encounter any issues, please report them by opening an issue. We will address any bugs promptly.

---

## Citation

```
@article{kang2024how,
  title={How Far is Video Generation from World Model? -- A Physical Law Perspective},
  author={Kang, Bingyi and Yue, Yang and Lu, Rui and Lin, Zhijie and Zhao, Yang, and Wang, Kaixin and Gao, Huang and Feng Jiashi},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}
```
