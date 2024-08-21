import os

def rename_files(folder_path):
    # Get the class name from the folder name
    class_name = os.path.basename(folder_path)
    
    # Path to the LANDMARKS folder
    landmarks_folder = os.path.join(folder_path, "LANDMARK")
    
    # Iterate through all files in the main folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Create the new filename for the image
            new_image_filename = f"{class_name}_{filename}"
            
            # Get the full file paths for the image
            old_image_file = os.path.join(folder_path, filename)
            new_image_file = os.path.join(folder_path, new_image_filename)
            
            # Rename the image file
            os.rename(old_image_file, new_image_file)
            print(f"Renamed image: {filename} -> {new_image_filename}")
            
            # Now handle the corresponding landmark file
            landmark_filename = os.path.splitext(filename)[0] + "_landmarks.json"
            new_landmark_filename = f"{class_name}_{landmark_filename}"
            
            old_landmark_file = os.path.join(landmarks_folder, landmark_filename)
            new_landmark_file = os.path.join(landmarks_folder, new_landmark_filename)
            
            # Check if the landmark file exists before renaming
            if os.path.exists(old_landmark_file):
                os.rename(old_landmark_file, new_landmark_file)
                print(f"Renamed landmark: {landmark_filename} -> {new_landmark_filename}")
            else:
                print(f"Warning: Landmark file not found for {filename}")

# Example usage
folder_path = r"D:\SignLanguage\MAIN_DATASET\Backup\RIGHT_NAMES\HANS\Z"
rename_files(folder_path)