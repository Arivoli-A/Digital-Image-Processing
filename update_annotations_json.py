import json
import argparse


def update_labels_to_bdd100k_format(input_json, output_json):
    """
    Update COCO labels to BDD100K standard format
    
    Your current format → BDD100K format:
    1: bicycle → 8: bicycle
    2: bus → 5: bus
    3: car → 3: car
    4: motorcycle → 7: motorcycle
    5: pedestrian → 1: pedestrian
    6: rider → 2: rider
    7: traffic light → 9: traffic light
    8: traffic sign → 10: traffic sign
    9: truck → 4: truck
    """
    
    # Mapping from your current IDs to BDD100K standard IDs
    OLD_TO_NEW_ID = {
        1: 8,   # bicycle
        2: 5,   # bus
        3: 3,   # car (stays same)
        4: 7,   # motorcycle
        5: 1,   # pedestrian
        6: 2,   # rider
        7: 9,   # traffic light
        8: 10,  # traffic sign
        9: 4,   # truck
    }
    
    # BDD100K standard categories
    BDD100K_CATEGORIES = [
        {"id": 1, "name": "pedestrian", "supercategory": "none"},
        {"id": 2, "name": "rider", "supercategory": "none"},
        {"id": 3, "name": "car", "supercategory": "none"},
        {"id": 4, "name": "truck", "supercategory": "none"},
        {"id": 5, "name": "bus", "supercategory": "none"},
        {"id": 6, "name": "train", "supercategory": "none"},
        {"id": 7, "name": "motorcycle", "supercategory": "none"},
        {"id": 8, "name": "bicycle", "supercategory": "none"},
        {"id": 9, "name": "traffic light", "supercategory": "none"},
        {"id": 10, "name": "traffic sign", "supercategory": "none"}
    ]
    
    print(f"Loading {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Original annotations: {len(data['annotations'])}")
    print(f"Original categories: {len(data['categories'])}")
    
    # Update categories to BDD100K standard (includes train even if no annotations)
    data['categories'] = BDD100K_CATEGORIES
    
    # Update all annotation category IDs
    updated_count = 0
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in OLD_TO_NEW_ID:
            ann['category_id'] = OLD_TO_NEW_ID[old_cat_id]
            updated_count += 1
        else:
            print(f"Warning: Unknown category ID {old_cat_id} in annotation {ann['id']}")
    
    # Save updated JSON
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Updated {updated_count} annotations")
    print(f"✓ Saved to {output_json}")
    print(f"\nNew categories:")
    for cat in BDD100K_CATEGORIES:
        print(f"  ID {cat['id']:2d}: {cat['name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO labels to BDD100K label IDs.")
    
    parser.add_argument("--input_json",
        default="test_annotations.json",
        help="Input JSON file (default: test_annotations.json)"
    )

    parser.add_argument("--output_json",
        default="test_annotations_bdd100k.json",
        help="Output JSON file (default: test_annotations_bdd100k.json)"
    )
    
    args = parser.parse_args()

    update_labels_to_bdd100k_format(args.input_json, args.output_json)
