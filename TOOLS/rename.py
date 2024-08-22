import os

def get_class_name(directory):
    # Extract the class name from the directory path
    return os.path.basename(directory)

def rename_files(directory, hand, name):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    class_name = get_class_name(directory)
    counter = 1

    # Rename files in the main directory
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            _, ext = os.path.splitext(filename)
            new_filename = f"{class_name}_{hand}_{name}_{counter}{ext}"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: {filename} -> {new_filename}")
            counter += 1

    # Rename files in the LANDMARKS subdirectory
    landmarks_dir = os.path.join(directory, 'LANDMARK')
    if os.path.exists(landmarks_dir):
        counter = 1  
        for filename in os.listdir(landmarks_dir):
            if os.path.isfile(os.path.join(landmarks_dir, filename)):
                new_filename = f"{class_name}_{hand}_{name}_{counter}.json"
                os.rename(os.path.join(landmarks_dir, filename), os.path.join(landmarks_dir, new_filename))
                print(f"Renamed in LANDMARKS: {filename} -> {new_filename}")
                counter += 1

    print("Renaming complete!")

# Set your parameters
directory = r"D:\SignLanguage\MAIN_DATASET\Backup\RIGHT_NAMES\XANDRA\Q"
HAND = "RIGHT"
NAME = "X"  

# Run the renaming function
rename_files(directory, HAND, NAME)