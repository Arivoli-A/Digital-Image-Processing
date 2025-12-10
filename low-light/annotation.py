import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == "__main__":
    # Load your images
    dataset = fo.Dataset.from_dir(
        dataset_dir="./ground_truth/test",
        dataset_type=fo.types.ImageDirectory,
    )
    dataset.persistent = True

    # Apply zero-shot detection model (YOLO-World or Grounding DINO)
    # Apply zero-shot detection model (YOLO-World)
    from ultralytics import YOLO

    # Load a pretrained YOLOv8s-worldv2 model
    model = YOLO("yolov8s-worldv2.pt")

    # Define custom classes
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
    ]
    model.set_classes(classes)

    # Apply the model to the dataset
    dataset.apply_model(model, label_field="predictions")

    # Launch the App
    session = fo.launch_app(dataset)
    
    # Example: Launch annotation in CVAT (requires Docker and setup)
    dataset.annotate(
        "annotation_run_1",
        label_field="predictions",
        backend="cvat",
        launch_editor=True,
    )
    
    session.wait()

    # Export annotations to JSON (COCO format)
    print("Exporting annotations to instances_updated.json...")
    dataset.export(
        export_dir=".",
        dataset_type=fo.types.COCODetectionDataset,
        labels_path="instances_updated.json",
        label_field="predictions",
    )
    print("Export complete.")
