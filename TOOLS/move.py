import os
import shutil

def copy_and_organize_files(source_path, dest_path):
    # Ensure the LANDMARKS folder exists in the destination
    landmarks_dest = os.path.join(dest_path, 'LANDMARKS')
    if not os.path.exists(landmarks_dest):
        print("Error: LANDMARKS folder does not exist in the destination.")
        return

    # Get the class name from the source path
    class_name = os.path.basename(source_path)

    # Create corresponding directory in destination for the class
    class_dest = os.path.join(dest_path, class_name)
    os.makedirs(class_dest, exist_ok=True)

    # Copy image files to the class folder
    for file in os.listdir(source_path):
        if file != 'LANDMARK':
            src = os.path.join(source_path, file)
            dst = os.path.join(class_dest, file)
            shutil.copy2(src, dst)

    # Copy files from LANDMARK folder to LANDMARKS
    landmark_path = os.path.join(source_path, 'LANDMARK')
    if os.path.exists(landmark_path):
        for file in os.listdir(landmark_path):
            src = os.path.join(landmark_path, file)
            dst = os.path.join(landmarks_dest, f"{file}")
            shutil.copy2(src, dst)

    print(f"Processed: {class_name}")

# Source and destination paths
source_path = r"D:\SignLanguage\MAIN_DATASET\Backup\LEFT_NAMES\CLOUD\Z"
dest_path = r"D:\SignLanguage\MAIN_DATASET\Backup\COMBINED"

# Run the reorganization
copy_and_organize_files(source_path, dest_path)

print("File reorganization complete!")