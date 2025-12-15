import os

def rename_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # optional: ensures consistent order

    for index, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1]  # keeps original extension
        new_name = f"rain_image_{index}{ext}"
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)

    print("Renaming completed.")

# 👉 Change this to your folder path:
rename_files("images_pre_cleaned")
