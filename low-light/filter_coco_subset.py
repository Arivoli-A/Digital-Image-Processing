import json
import re
import argparse

def extract_i_j(filename):
    """
    Extract (i,j) from filenames, ignoring optional '.rf.<hash>'
    """
    match = re.search(r"ucsd_night(\d+)_frame_0*(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None




def merge_and_filter_train_test(
    train_json,
    test_json,
    tuple_list,
    output_json
):
    ij_set = set(tuple_list)

    # ---- Load JSON files ----
    with open(train_json, "r") as f:
        train_data = json.load(f)

    with open(test_json, "r") as f:
        test_data = json.load(f)

    # ---- Merge data ----
    all_images = train_data["images"] + test_data["images"]
    all_annotations = train_data["annotations"] + test_data["annotations"]

    # categories should be identical → take from train
    categories = train_data["categories"]

    # ---- Filter images ----
    kept_images = []
    old_to_new_img_id = {}
    seen_ij = set()
    new_img_id = 1

    for img in all_images:
        ij = extract_i_j(img["file_name"])

        # keep only selected (i, j), avoid duplicates
        if ij in ij_set and ij not in seen_ij:
            seen_ij.add(ij)
            old_to_new_img_id[img["id"]] = new_img_id
            img["id"] = new_img_id
            kept_images.append(img)
            new_img_id += 1

    valid_old_img_ids = set(old_to_new_img_id.keys())

    # ---- Filter annotations ----
    kept_annotations = []
    new_ann_id = 1

    for ann in all_annotations:
        if ann["image_id"] in valid_old_img_ids:
            ann["image_id"] = old_to_new_img_id[ann["image_id"]]
            ann["id"] = new_ann_id
            kept_annotations.append(ann)
            new_ann_id += 1

    # ---- Final COCO ----
    final_coco = {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(final_coco, f, indent=2)

    print("✓ Merge + filter complete")
    print(f"✓ Images kept      : {len(kept_images)}")
    print(f"✓ Annotations kept: {len(kept_annotations)}")
    print(f"✓ Saved to         : {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge train & test COCO JSONs and filter using (i, j) tuple list."
    )

    parser.add_argument("--train_json", required=True, help="Train COCO JSON")
    parser.add_argument("--test_json", required=True, help="Test COCO JSON")
    parser.add_argument("--output_json", required=True, help="Output merged JSON")

    args = parser.parse_args()

    
    # 🔹 PUT YOUR (i, j) TUPLES HERE 🔹
    IJ_LIST =  [
        (2, 7), (2, 9), (2, 10), (2, 12), (2, 18),
        (3, 10),
        (4, 5), (4, 6), (4, 8), (4, 9), (4, 10), (4, 22), (4, 36), (4, 37),
        (5, 2), (5, 9), (5, 10), (5, 14), (5, 36), (5, 43), (5, 44), (5, 48), (5, 60), (5, 67), (5, 70)
    ]
    

    merge_and_filter_train_test(
        args.train_json,
        args.test_json,
        IJ_LIST,
        args.output_json
    )
