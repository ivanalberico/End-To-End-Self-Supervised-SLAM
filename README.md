# End-To-End-Self-Supervised-SLAM

## Requirements

### Installation gradslam (https://github.com/gradslam/gradslam)

Using `pip` (Experimental)

`pip install gradslam`

Install from GitHub

`pip install 'git+https://github.com/gradslam/gradslam.git'`

Install from local clone (recommended)
```
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
pip install .
cd ..
git clone https://github.com/gradslam/gradslam.git
cd gradslam
pip install -e .[dev]
```

### PyTorch 
It is recommended to install the latest version, so that we all works on the same version.

`pytorch >= 1.6.0`
`CUDA >= 10.1`

### Python
`python >= 3.7.0`

## Files
`absolute_scale.py`: This file will train a scaling layer for depth maps, consisting of an affine transformation (scale + offset) (Additional Works B.2) 

`demo.py`: This file contains the script used to make the demo for the final presentation (Demo - Final Presentation) 

`gradient_experiments.py`: This file contains the gradient experiments carried out during the early phase of the project (Additional Works Appendix B.1) 

`median_scaling.py`: This file computes the median scale for an entire dataset (Improvement / Debugging) 

`online_adaption.py`: This file contains the final working model that contains the online adaption module integrated with SLAM reconstruction (Final Working SLAM system) 

`pose_checker.py`: This file contains a simple script to to compute transformation between two specified poses (Debugging)

`test_depth_scaling.py`: This script is used for obtaining evaluation results for the learned scaling parameters (Additional Works B.2). Basically runs the online refinement module and saves depth predictions to disk.

`train_depth.py`: This file contains the developement of online adaption module. Included are numerous experiments controlled behind different flags that can be controlled by the config file. 

`train_depth_OFT.py`: This file contains the idea of output finetuning but was not used in our final system. (Improvements)

## Datasets

GradSlam provides useful tools and functions to download the datasets in a suitable format. For a detailed explanation follow the instructions and example script in this [link](https://gradslam.readthedocs.io/en/latest/tutorials/tutorial_prerequisits.html). See below two examples on how to load a trajectory for each (ICL, TUM) dataset:

### ICL

```
# download 'lr kt1' of ICL dataset
if not os.path.isdir('ICL'):
    os.mkdir('ICL')
if not os.path.isdir('ICL/living_room_traj1_frei_png'):
    print('Downloading ICL/living_room_traj1_frei_png dataset...')
    os.mkdir('ICL/living_room_traj1_frei_png')
    !wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz -P ICL/living_room_traj1_frei_png/ -q
    !tar -xzf ICL/living_room_traj1_frei_png/living_room_traj1_frei_png.tar.gz -C ICL/living_room_traj1_frei_png/
    !rm ICL/living_room_traj1_frei_png/living_room_traj1_frei_png.tar.gz
    !wget https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom1n.gt.sim -P ICL/living_room_traj1_frei_png/ -q
    print('Downloaded.')
icl_path = 'ICL/'
```

### TUM

```
# download 'freiburg1' of TUM dataset
if not os.path.isdir('TUM'):
    os.mkdir('TUM')
if not os.path.isdir('TUM/rgbd_dataset_freiburg1_xyz'):
    print('Downloading TUM/rgbd_dataset_freiburg1_xyz dataset...')
    !wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz -P TUM/ -q
    !tar -xzf TUM/rgbd_dataset_freiburg1_xyz.tgz -C TUM/
    !rm TUM/rgbd_dataset_freiburg1_xyz.tgz
    print('Downloaded.')
tum_path = 'TUM/'
```

Please put both dataset in a folder under the respective dataset names since the code assume this.

## Running the code!

Nearly every script relies on the `config/config.yaml` in order to set the required parameters. 

The main files to test our results are as following:

`train_depth.py` will run the online adaption model on a pair of key-frames. The model can be trained on additional key-frames by changing the `DEBUG:iter_stop` variable in the config file. However, in order to test the performance of the online adaption model, we recommend to stick with just a pair.

The results presented in the report can be reproduced by setting the following flags in the `config/config.yaml`:

### ICL
```
DATA:
  name: ICL
  data_path: path\to\data
  dilation: 2
  start: 418
```

### TUM
```
DATA:
  name: TUM
  data_path: path\to\data
  dilation: 5
  start: 115
```

Please do not change any other flags to ensure reproducibility of the exact frames. In the train_depth.py the median scale is pre-computed per scene therefore training on other scenes might not be as desired. 

Then run the command: 
```
python3 train_depth.py --config_path path\to\config
```



### How to find median scale for a particular frame
```
DATA:
  name: ICL
  data_path: path\to\data
  start: Set which frame to start from
```

Then run the command:

```
python3 median_scaling.py --config_path path\to\config
```

### Online Adaption + PointFusion (Final Model)
In order to see the full system in action, please use the `online_adaption.py` script provided.

First set the config file appropriately for the dataset:
```
DATA:
  name: ICL
  data_path: path\to\data
  dilation: 5
  start: 0
OPTIMIZATION:
  refinement_steps: 3     # We recommend only 2-3 refinement steps per pair of key-frames to avoid overfitting.
DEMO:
  sequence_length: 60     # Set the number of frames in the entire sequence you want to reconstruct since we will load them on the GPU together
  frame_threshold: 0.05   # This variable needs to be adjusted according to datasets (0.05 for ICL, 0.12 for TUM)
```
There is no need to set median scaling here since its computed on the fly automatically. The `config.yaml` additional provides a variety of losses that can be incorporated in the online adaption module, however, we realize that a simple approach of photometric loss + end-2-end point supervision + depth regularization works well. 

To the run online adaption module:
```
python3 online_adaption.py --config_path path\to\config
```
Note that due to scale inconsistencies (even after median scaling) the global pointcloud results might not be as imperessive overlayed on top of each other.

### Unsupervised Scale Learning

Multiple parameters for the training procedure are defined in `configs/config_scale_learning.yaml`, and can be adjusted as described in the previous sections.

To run the scale learning procedure use the following command:
```
python3 absolute_scale.py --config_path configs/config_scale_learning.yaml
```
**NOTE**: When training please set the all the boolean parameters in the `ABLATION` section of the `configs/config_scale_learning.yaml` script to **False**.

Once the training phase is finished, the learned weight and bias can be added to the `configs/config_scale_learning.yaml` in the following fields
```
ABLATION:
  ...
  scaled_depth: True
  scaling_depth: 6.0891 
  with_bias: True
  bias: -1.0958
```
Set the corresponding flags for the scale and bias to **True** and run the following command for evaluating the learned parameters during online refinement:

```
python3 test_depth_scaling.py --config_path configs/config_scale_learning.yaml
```

### Demo

The demo contains good visualization functions:

```
DATA:
  name: ICL
  data_path: path\to\data
  dilation: 5
  start: 0
OPTIMIZATION:
  refinement_steps: 3     # We recommend only 2-3 refinement steps per pair of key-frames to avoid overfitting.
DEMO:
  sequence_length: 60     # Set the number of frames in the entire sequence you want to reconstruct since we will load them on the GPU together
  frame_threshold: 0.05   # This variable needs to be adjusted according to datasets (0.05 for ICL, 0.12 for TUM)
```

To the run the demo:

```
python3 demo.py --config_path path\to\config
```

## Usage

### Depth Map Masking on TUM Dataset

The ground-truth depth maps in the TUM dataset are incomplete, containing multiple pixels with zero measurements. This can become an issue in computing loss metrics for supervision of the refinement process. 
To exclude these measurements from the loss computations, a simple mask is implemented that excludes any zero value from the computation of loss metrics. It is included behind the flag `tum_depth_masking` in the configuration file.

To enable depth map masking for the TUM dataset, change the file at `configs/config.yaml`:

```
LOSS:
  tum_depth_masking: True
```

### Gradient Visualizations with Tensorboard

Getting insights into which parts of the model are activated at different stages of the training and refinement process is valuable for supervision and debugging purposes. 
Outputs for gradient histograms and last layer image representation can be requested during the training. To enable it change the following settings in the config file at `configs/config.yaml`:

```
VIZ:
  tensorboard: True
```

This will register hook functions on all convolutional layers of the respective decoders and works for both `monodepth2` and the `indoor` pretrained model. By default, all values are returned as absolute measurements. For the image representation of the gradients, the output can be scaled between minimum and maximum values for each refinement step. This is done via the following flag:

```
VIZ:
  tensorboard_scaled: True
```

NOTE: as the methodology described above accesses the gradients of the respective layers, it can only be used with `autograd` enabled. Therefore the visualization tools are not available when running the `train_depth_OFT.py` script.
