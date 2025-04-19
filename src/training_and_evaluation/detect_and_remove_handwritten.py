import os
import sys
from utils import mark_all_handwritten,separate_handwritten

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)


# for now we only do it for the training_dataset 
# TODO : maybe we can do it for the validation dataset later as well
handwritten_images = mark_all_handwritten('./Datasets/original/train_deepl.csv')


# copy the handwritten to a new temporary folder so we can inspect them 
# and decide which ones we really want to drop (optional)
separate_handwritten(handwritten_images,'./handwritten_images')
