import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == "__main__":
    # Load your images
    dataset = fo.Dataset.from_dir(
        dataset_dir="./rain-removal/images_pre_cleaned",
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
        "pedestrian",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "rider",
        "traffic light",
        "traffic sign",
    ]
    model.set_classes(classes)

    # Apply the model to the dataset
    dataset.apply_model(model, label_field="predictions")

    # Launch the App
    session = fo.launch_app(dataset)
    
   # ... (Previous code up to annotation launch)

    # 1. Launch annotation in CVAT
    cvat_job = dataset.annotate(
        "annotation_run_1",
        label_field="predictions",
        backend="cvat",
        launch_editor=True,
    )
    
    session.wait() # This waits for the FiftyOne App to close, which is fine.
                   # You need to manually wait for the CVAT job to finish in the CVAT UI.
    
    # --- V V V ADD THIS CRITICAL STEP V V V ---
    
    # 2. WAIT FOR THE CVAT JOB TO COMPLETE (Manual step)
    #    You must manually finish/submit the task inside the CVAT web interface.
    #    Once completed, the job status will change.
    
    # 3. PULL the completed annotations back into the FiftyOne dataset
    print("Waiting for CVAT job to be completed and importing annotations...")
    cvat_job.wait_until_done()
    cvat_job.cleanup() # This imports the annotations back into the 'predictions' field
    
    # --- ^ ^ ^ ADD THIS CRITICAL STEP ^ ^ ^ ---
    
    # 4. Export annotations to JSON (COCO format)
    print("Exporting annotations to instances_updated.json...")
    dataset.export(
        export_dir=".",
        dataset_type=fo.types.COCODetectionDataset,
        labels_path="instances_updated.json",
        label_field="predictions", # This field now contains the human-validated data
    )
    print("Export complete.")