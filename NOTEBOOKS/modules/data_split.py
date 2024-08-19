import os
import shutil
from sklearn.model_selection import train_test_split


"""
SPLITTED_DATASET = "DATASET"
PATH = "path/to/your/source/directory"

# Create an instance of DatasetSplitter
splitter = DatasetSplitter(PATH, SPLITTED_DATASET)

# Use the default 70-30 split
splitter()

# Or specify a different split, e.g., 80-20
splitter(train_size=0.8)

# You can also call split_dataset directly if you prefer
splitter.split_dataset(train_size=0.75)
"""

class DatasetSplitter:
    def __init__(self, source_dir, output_dir, random_state=42):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.random_state = random_state
        self.train_dir = os.path.join(output_dir, 'train')
        self.test_dir = os.path.join(output_dir, 'test')
        self.image_extensions = ('jpg', 'jpeg', 'png', 'heic')

    def create_output_dirs(self):
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def get_images(self, subdir_path):
        return [os.path.join(subdir_path, img) for img in os.listdir(subdir_path) 
                if img.lower().endswith(self.image_extensions)]

    def copy_images(self, images, target_dir):
        for img in images:
            print(f"COPYING {img} TO {target_dir}")
            shutil.copy(img, os.path.join(target_dir, os.path.basename(img)))

    def split_subdir(self, subdir, train_size):
        subdir_path = os.path.join(self.source_dir, subdir)
        images = self.get_images(subdir_path)
        
        print(f"Found {len(images)} images in {subdir_path}")

        if len(images) < 1:
            print(f"No images found in {subdir_path}, skipping...")
            return

        train_images, test_images = train_test_split(
            images, train_size=train_size, shuffle=True, random_state=self.random_state
        )
        
        print(f"Splitting {len(images)} images: {len(train_images)} for training and {len(test_images)} for testing in {subdir_path}")
        print("--------------------------------------------------------------------")

        train_subdir = os.path.join(self.train_dir, subdir)
        test_subdir = os.path.join(self.test_dir, subdir)

        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)
        
        self.copy_images(train_images, train_subdir)
        self.copy_images(test_images, test_subdir)
        
        print(f"SPLITTING DONE FOR {subdir}, DATASET SAVED TO: {self.output_dir}")

    def split_dataset(self, train_size=0.7):
        self.create_output_dirs()
        for subdir in next(os.walk(self.source_dir))[1]:
            self.split_subdir(subdir, train_size)

    def __call__(self, train_size=0.7):
        self.split_dataset(train_size)