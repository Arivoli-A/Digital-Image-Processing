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

 ## Rain Suppression

  This module implements Wavelet Transform and Morphological Component Analysis approaches to remove rain occlusions from images, improving object detection performance in rainy conditions.

  ### Setting Up the Dataset

  1. Download the `images_pre_processed.tar.gz` and `rain_annotations.json` from [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1rKZmiwryIud3RELx5n39hYT4YGhbBqlf).

  2. Extract the `images_pre_processed.tar.gz` into `rain-removal/images_pre_processed`:
     ```bash
     cd rain-removal
     tar -xzf images_pre_processed.tar.gz

  3. Place the rain_annotations.json file in the rain-removal directory.

  Running Rain Removal Experiments

  Step 1: Preprocess Images

  First, run the Wavelet Filtering or MCA processing on the images in the rain dataset:

  1. Open rain-removal/rain_removal_optimized.ipynb
  2. Choose your processing method (Wavelet or MCA)
  3. Run all cells in the notebook

  This will create processed image directories:
  - inputs_images_processed_wavelet/ for Wavelet processing
  - inputs_images_processed_mca/ for MCA processing

  Step 2: Evaluate Model Performance

  Evaluate the Faster R-CNN model on both unprocessed and processed images:

  Basic usage:
  # Evaluate with baseline BDD100K model on Wavelet-processed images
  python evaluation_a_rain.py wavelet

  # Evaluate with baseline BDD100K model on MCA-processed images
  python evaluation_a_rain.py mca

  Using the fine-tuned model:
  # Evaluate with rain fine-tuned model on Wavelet-processed images
  python evaluation_a_rain.py wavelet --model finetuned

  # Evaluate with rain fine-tuned model on MCA-processed images
  python evaluation_a_rain.py mca --model finetuned

  Arguments:
  - processed_type (required): Choose wavelet or mca
  - --model: Choose baseline (default) or finetuned

  Output:
  - Results saved to evaluation_results/
  - Visualized predictions saved to output_unprocessed_images/ and output_processed_{type}/
  - Metrics: AP, AP50, AP75, AR

  Step 3: Fine-tune the Model (Optional)

  Fine-tune the Faster R-CNN model on the rain dataset:

  Basic usage with default settings:
  python finetune_rcnn_rain.py

  Custom training configuration:
  python finetune_rcnn_rain.py \
    --images ./images_pre_processed \
    --annotations ./rain_annotations.json \
    --iterations 5000 \
    --batch-size 4 \
    --learning-rate 0.0005

  Available arguments:
  - --images: Path to training images (default: ./images_pre_processed)
  - --annotations: Path to annotations JSON (default: ./rain_annotations.json)
  - --base-checkpoint: Base model to start from (default: ../FasterRCNN/finetune_bdd.pth)
  - --output-dir: Directory for training outputs (default: ../FasterRCNN/rain_finetuned)
  - --output-model-name: Final model filename (default: finetune_rain.pth)
  - --batch-size: Training batch size (default: 2)
  - --learning-rate: Learning rate (default: 0.00025)
  - --iterations: Maximum training iterations (default: 3000)
  - --val-split: Validation split ratio (default: 0.2)
  - --checkpoint-period: Save checkpoint every N iterations (default: 500)
  - --eval-period: Evaluate every N iterations (default: 300)

  Output:
  - Training checkpoints saved to ../FasterRCNN/rain_finetuned/
  - Final model saved as ../FasterRCNN/finetune_rain.pth
  - Training logs and validation results displayed during training

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
