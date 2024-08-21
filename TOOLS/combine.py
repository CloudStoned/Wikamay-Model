import os
import shutil

# Define the paths
base_path = r"D:\SignLanguage\MAIN_DATASET\Backup\SORTED"
left_path = os.path.join(base_path, "LEFT")
right_path = os.path.join(base_path, "RIGHT")
combine_path = os.path.join(base_path, "COMBINED")

# Create the COMBINE folder if it doesn't exist
os.makedirs(combine_path, exist_ok=True)
print(f"Created or verified COMBINE folder: {combine_path}")

def copy_files_with_structure(src_folder, dst_folder):
    print(f"\nProcessing folder: {src_folder}")
    for class_folder in os.listdir(src_folder):
        class_path = os.path.join(src_folder, class_folder)
        if os.path.isdir(class_path):
            dst_class_path = os.path.join(dst_folder, class_folder)
            os.makedirs(dst_class_path, exist_ok=True)
            print(f"\n  Created or verified class folder: {dst_class_path}")
            
            # Create LANDMARKS folder
            landmarks_folder = os.path.join(dst_class_path, "LANDMARKS")
            os.makedirs(landmarks_folder, exist_ok=True)
            print(f"  Created or verified LANDMARKS folder: {landmarks_folder}")
            
            # Copy files
            for item in os.listdir(class_path):
                item_path = os.path.join(class_path, item)
                if os.path.isfile(item_path):
                    # Copy image file to class folder
                    dst_file = os.path.join(dst_class_path, item)
                    print(f"    Copying image file: {item} to {dst_file}")
                    shutil.copy2(item_path, dst_file)
                elif item == "LANDMARKS":
                    # Copy contents of LANDMARKS folder
                    src_landmarks = os.path.join(class_path, item)
                    for landmark_file in os.listdir(src_landmarks):
                        src_file = os.path.join(src_landmarks, landmark_file)
                        dst_file = os.path.join(landmarks_folder, landmark_file)
                        print(f"    Copying landmark file: {landmark_file} to {dst_file}")
                        shutil.copy2(src_file, dst_file)

# Copy files from LEFT and RIGHT folders
print("Starting to copy files from LEFT folder")
copy_files_with_structure(left_path, combine_path)

print("\nStarting to copy files from RIGHT folder")
copy_files_with_structure(right_path, combine_path)

print("\nFiles have been combined successfully with the desired structure.")