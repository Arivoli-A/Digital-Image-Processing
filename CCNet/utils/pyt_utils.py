# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist

from .logger import get_logger

logger = get_logger()

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device, weights_only=False)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Remap BatchNorm keys: checkpoint was saved with inplace_abn (no .0), 
    # but our Sequential fallback expects .0 (e.g., bn1.weight -> bn1.0.weight)
    model_state = model.state_dict()
    remapped_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Check if this key exists in model as-is
        if k in model_state and model_state[k].shape == v.shape:
            remapped_state_dict[k] = v
        else:
            # Try adding .0 for BatchNorm parameters
            # Pattern 1: ...bn1.weight -> ...bn1.0.weight (for bn1, bn2, bn3)
            # Pattern 2: ...head.conva.1.weight -> ...head.conva.1.0.weight (for Sequential modules)
            parts = k.split('.')
            remapped = False
            for i, part in enumerate(parts):
                # Pattern 1: bn1, bn2, bn3 followed by BatchNorm param
                if part in ['bn1', 'bn2', 'bn3'] and i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        new_key = '.'.join(parts[:i+1] + ['0'] + parts[i+1:])
                        if new_key in model_state and model_state[new_key].shape == v.shape:
                            remapped_state_dict[new_key] = v
                            remapped = True
                            break
                # Pattern 2: Sequential module (numeric index) followed by BatchNorm param
                # e.g., head.conva.1.weight -> head.conva.1.0.weight
                if part.isdigit() and i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        new_key = '.'.join(parts[:i+1] + ['0'] + parts[i+1:])
                        if new_key in model_state and model_state[new_key].shape == v.shape:
                            remapped_state_dict[new_key] = v
                            remapped = True
                            break
            
            if not remapped:
                # If no remapping worked, keep original (might be used or ignored)
                remapped_state_dict[k] = v
    
    model.load_state_dict(remapped_state_dict, strict=False)
    ckpt_keys = set(remapped_state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    # Filter out num_batches_tracked - it's only used during training, not needed for inference
    missing_keys_filtered = [k for k in missing_keys if 'num_batches_tracked' not in k]
    
    if len(missing_keys_filtered) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys_filtered)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    del remapped_state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def decode_predictions(preds, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
      pixels = img.load()
      for j_, j in enumerate(preds[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1,2,0)) + img_mean).astype(np.uint8)
    return outputs
