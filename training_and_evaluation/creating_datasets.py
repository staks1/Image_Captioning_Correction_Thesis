import os
import sys

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_1 ='/home/ekotsis/s4a/training_and_evaluation/Models/zeroshot_clip/original/embeddings/test/'


# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

from config_files import *
from fix_images_paths import *


# create the directories where we will save the image caption pairs (for original and augmented datasets)
# change this to your own paths
original_path = "./Datasets/original/"
augmented_path = "./Datasets/augmented/"


# create original/trainimages - valimages - testimages image caption pairs
# inside the original directory
# TODO : we will need to do the same for the augmented dataset inside the augmented directory
# when we want to create the augmented datasets

if not (os.path.exists(os.path.join(original_path))):
    os.makedirs(original_path)

if not (os.path.exists(os.path.join(augmented_path))):
    os.makedirs(augmented_path)


# create the models directory {zeroshot_clip,clip_trained_projection_layers,clip_trained_transformers_last,clip_with_classification_projection_layers,clip_with_classification_transformers_last}


paths = []

root_model_dir = "Models"
root_dirs = [
    "zeroshot_clip",
    "clip_trained_projection_layers",
    "clip_trained_transformers_last",
    "clip_with_classification_projection_layers",
    "clip_with_classification_transformers_last",
]

subfolders = ["original", "augmented"]
main_folders = ["weights", "embeddings", "results", "plots_misc"]
embedding_subfolders = ["train", "val", "test"]

for root in root_dirs:
    for subfolder in subfolders:
        for main_folder in main_folders:
            if main_folder == "embeddings":
                for embed_subfolder in embedding_subfolders:
                    paths.append(
                        f"{root_model_dir}/{root}/{subfolder}/{main_folder}/{embed_subfolder}"
                    )
            else:
                paths.append(f"{root_model_dir}/{root}/{subfolder}/{main_folder}")

# Create the directories, skipping if they already exist
for path in paths:
    os.makedirs(path, exist_ok=True)

print("=" * 100)
print("All paths created")
print("=" * 100)
