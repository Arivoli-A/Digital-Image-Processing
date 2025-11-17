## Custom datasets

Drop any evaluation-only imagery inside sub-folders under this directory.  
Suggested layout:

```
datasets/
└── custom/
    ├── images/          # raw RGB inputs (png/jpg/jpeg)
    └── outputs/         # optional location for predictions
```

The inference helper at `tools/run_inference.py` accepts any folder path via
`--image-dir` so you can create additional sub-folders (e.g. `datasets/city`,
`datasets/indoor`). Images are automatically resized, normalized with the
Cityscapes mean and fed through the CCNet model.

