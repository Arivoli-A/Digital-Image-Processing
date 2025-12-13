# Digital-Image-Processing
Digital Image Processing for Autonomous Vehicles

## Low-Light Enhancement

This module implements low-light image enhancement techniques such as **BIMEF** and **LIME** to improve visibility in night-time or poorly lit images.

### Setup

1. **Create and activate the conda environment**:

```bash
bash conda_setup.sh
conda activate opencv_env
```

### Running Low-Light Enhancement Experiments

The low-light enhancement experiments are located in the `low-light` folder, within the `Low_Light_experiment.ipynb` notebook. To run the experiments, select the `opencv_env` kernel inJupyter Notebook.


- **LIME**: Python implementation is available in this repository, adapted from [Python-LIME](https://github.com/wiitt/Python-LIME). A C++ version is also explored: [LIME_Processing](https://github.com/zj611/LIME_Processing.git).  
- **BIMEF**: Available via OpenCV under the `intensity_transform` submodule.


### Object Detection Model

The Faster R-CNN model has been fine-tuned for low-light images.  

**To fine-tune the model:**

```bash
python finetune_rcnn_low_light.py
```

**To evaluate the model on low-light images:**

```bash
python evaluation_a_low_light.py
```

This will generate performance metrics and outputs for both enhanced and unprocessed low-light images.
