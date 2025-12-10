# . ~/.bashrc
# conda activate opencv_env

python ../FasterRCNN/run_inference.py --image-dir ground_truth/train --output-dir outputs/ground_truth \
--annotation ground_truth/train/instances_default_train.json # --thing-classes objects Bus Car Light Person Sign Truck --num-classes 7 

# python ../FasterRCNN/train_custom.py \
#     --train-json ground_truth/train/_annotations.coco.json \
#     --train-images ground_truth/train/train \
#     --num-classes 6 \
#     --thing-classes objects Bus Car Light Person Sign Truck \
#     --max-iter 1000


# import json
# with open(args.annotation) as f:
#     ann = json.load(f)
# print("Annotation category IDs:", [c["id"] for c in ann["categories"]])