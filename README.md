# Digital-Image-Processing
Digital Image Processing for Autonomous Vehicles

## Setup

1. **Create and activate the conda environment**:

```bash
bash conda_setup.sh
conda activate opencv_env
```

2. Download the model weights `untuned_model.pth` from [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1H_2U9atsOXpjMnzDovqmYGdlFC5n7mvb). Put the `untuned_model.pth` file into the `FasterRCNN` directory.

If you want to annotate any of the images, update the image directories in `annotation.py` and then run it.

## Fog Removal

This module implements **Dark Channel Prior (DCP)** and a pipeline which combines **DCP** and **DWT/CLAHE** from the [2018 Kim et al. paper](https://doi.org/10.1049/iet-ipr.2016.0819) to remove fog from images.

### Setting Dataset Up

1. Download the `fog_data.zip` and `labels.json` from [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1rKZmiwryIud3RELx5n39hYT4YGhbBqlf). 
   
2. Extract the `fog_data.zip` into `fog-removal/fog_dataset/input_images/unprocessed_images`. 
   
3. Put the `labels.json` in `fog-removal/fog_dataset`.

### Running Fog Removal Experiments

1. The first thing you need to do is preprocess the unprocessed images using the **DCP** and **DCP/DWT/CLAHE** pipelines. To do this, run the following commands:
```bash
cd fog-removal
python image_preprocessing.py
```

2. To run the experiment that compares the performance of the untuned FasterRCNN model on the unprocessed and processed images, run the following command:
```bash
python eval_image_processing_fog.py
```

3. To run the experiment that finetunes the FasterRCNN model and then compares it to the results from step 2, run the following command:
```bash
python eval_finetuning_fog.py
```

All of the results from the experiments are saved in the `evaluation_results/fog` folder.

## Rain Supression

This module implements Wavelet Transform and a Morphilogical Component Analysis Approach which can be used to remove rain occlusions from images.


### Setting Dataset Up

1. Download the `images_pre_processed.tar.gz` and `rain_annotations.json` from [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1rKZmiwryIud3RELx5n39hYT4YGhbBqlf). 
   
2. Extract the `images_pre_processed.tar.gz` into `rain-removal/images_pre_processed`. 
   
3. Put the `rain_annotations.json` in the main directory.

### Running Fog Removal Experiments

1. First, you must run the Wavelet Filtering or MCA processing on the images in the rain dataset. Do this by going into ```rain-removal/rain_removal_optimized.ipynb``` , choosing the processing method, and running the notebook.


2. To evaluate the performance of the detectron model on both the unprocessed and processed sets of images, run this command.
```bash
python eval_image_processing_fog.py
```

3. To run the experiment that finetunes the FasterRCNN model and then compares it to the results from step 2, run the following commands:
```bash
python evaluation_a_rain.py
```

All of the results from the experiments are saved in the `rain-removal/evaluation_results` folder.

## Low-Light Enhancement

This module implements low-light image enhancement techniques such as **BIMEF** and **LIME** to improve visibility in night-time or poorly lit images.


### Running Low-Light Enhancement Experiments

The low-light enhancement experiments are located in the `low-light` folder, within the `Low_Light_experiment.ipynb` notebook. To run the experiments, select the `opencv_env` kernel in Jupyter Notebook.

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
