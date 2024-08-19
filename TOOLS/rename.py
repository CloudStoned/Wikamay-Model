import os

def rename_files(root_dir):
    # List of subdirectories to process
    subdirs = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # Process IMG folder
        img_dir = os.path.join(subdir_path, 'IMG')
        if os.path.isdir(img_dir):
            rename_files_in_folder(img_dir)
        
        # Process LANDMARKS folder
        landmarks_dir = os.path.join(subdir_path, 'LANDMARK')
        if os.path.isdir(landmarks_dir):
            rename_files_in_folder(landmarks_dir)

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.startswith('RIGHT_'):  # Skip files that are already renamed
            old_path = os.path.join(folder_path, filename)
            new_filename = f"RIGHT_{filename}"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# Usage
root_directory = r"D:\SignLanguage\TOOLS\LETTERS_NUMS_LR\RIGHT_NAMES\XANDRA"
rename_files(root_directory)