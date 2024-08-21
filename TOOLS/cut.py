import os
import shutil

def reorganize_folder(base_path):
    # Iterate through all items in the base path
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            img_folder = os.path.join(item_path, 'IMG')
            
            # Check if IMG folder exists
            if os.path.exists(img_folder):
                # Move all files from IMG folder to parent folder
                for file in os.listdir(img_folder):
                    src = os.path.join(img_folder, file)
                    dst = os.path.join(item_path, file)
                    shutil.move(src, dst)
                
                # Remove the empty IMG folder
                os.rmdir(img_folder)
                print(f"Reorganized: {item_path}")
            else:
                print(f"IMG folder not found in: {item_path}")

# Path to the HANS folder
base_path = r"D:\SignLanguage\MAIN_DATASET\Backup\RIGHT_NAMES\XANDRA"

# Run the reorganization
reorganize_folder(base_path)

print("Reorganization complete!")