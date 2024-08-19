import os
import shutil
import string

def is_valid_class(class_name):
    valid_classes = set(str(i) for i in range(11)) | set(string.ascii_uppercase)
    return class_name in valid_classes

def reorganize_files(source_root, destination_root):
    print(f"Starting file reorganization...")
    print(f"Source directory: {source_root}")
    print(f"Destination directory: {destination_root}")

    # Walk through the source directory
    for root, dirs, files in os.walk(source_root):
        # Get the relative path from the source root
        rel_path = os.path.relpath(root, source_root)
        parts = rel_path.split(os.sep)

        # Skip the root directory itself
        if len(parts) == 1 and parts[0] == '.':
            continue

        # The class name is the first subdirectory under source_root
        class_name = parts[0]

        # Skip if not a valid class
        if not is_valid_class(class_name):
            print(f"Skipping invalid class: {class_name}")
            continue

        # Set up destination directories
        class_dir = os.path.join(destination_root, class_name)
        landmark_dir = os.path.join(class_dir, 'LANDMARKS')

        # Ensure class directory exists
        os.makedirs(class_dir, exist_ok=True)

        # Copy files based on their type
        for file in files:
            src_file = os.path.join(root, file)
            if 'LANDMARK' in root.upper():
                # This is a landmark file
                dst_file = os.path.join(landmark_dir, file)
                os.makedirs(landmark_dir, exist_ok=True)
            else:
                # This is an image file
                dst_file = os.path.join(class_dir, file)
            
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
            else:
                print(f"Skipped (already exists): {dst_file}")

    print("File reorganization complete.")

if __name__ == "__main__":
    source_root = r"D:\SignLanguage\TOOLS\LETTERS_NUMS_LR\RIGHT_NAMES\XANDRA"
    destination_root = r"D:\SignLanguage\TOOLS\LETTERS_NUMS_LR\DATASET\RIGHT"   

    reorganize_files(source_root, destination_root)