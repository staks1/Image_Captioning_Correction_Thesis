import os
import sys

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

from config_files import *
from fix_images_paths import *
from creating_datasets import *


if __name__ == "__main__":
    # create the directories for the embeddings files
    # this is where we are going to save the embeddings for different models,evaluations

    # generate the original image/caption pairs and jsonl in the appropriate folders
    a = 1
    # just for testing
    # in the training later we will make an args parser
    # also i add the category by default for all the datasets
    if a == 1:

        # optionally here we can apply the handwritten image caption pairs removal
        dp = DatasetsPreprocessor("train_deepl.csv")
        dp.fix_image_paths(path=original_path)
        dp.copy_images(path=original_path, add_category=True)

        dp2 = DatasetsPreprocessor("val_deepl.csv")
        dp2.fix_image_paths(path=original_path)
        dp2.copy_images(path=original_path, add_category=True)

        dp3 = DatasetsPreprocessor("test_deepl.csv")
        dp3.fix_image_paths(path=original_path)
        dp3.copy_images(path=original_path, add_category=True)

        print("=" * 100)
        print("Original Dataset's image-caption pairs were created")
        print("=" * 100)
    
    else:
        print("Skipped creating original dataset, Already created\n")
        #print("Skipped creating augmented dataset, Already created")