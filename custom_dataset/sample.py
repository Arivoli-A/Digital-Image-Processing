import cv2
import os

def extract_frames(video_path, filename, output_dir="raw_images", interval_sec=5):
   
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("Cannot get FPS for the video.")

    frame_interval = int(fps * interval_sec)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            # Save image
            img_name = os.path.join(output_dir, f"{filename}_frame_{saved_count:05d}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved: {img_name}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print("Done.")

if __name__ == "__main__":
    raw_data_dir = './raw_dataset'
    mp4_files = [f for f in os.listdir(raw_data_dir)
             if os.path.isfile(os.path.join(raw_data_dir, f)) and f.lower().endswith('.mp4')]

    for file in mp4_files:
        file_name = os.path.splitext(file)[0]
        extract_frames(raw_data_dir+ '/' + file, file_name)
    