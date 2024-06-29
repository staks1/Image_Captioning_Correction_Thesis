import os
import sys
from utils import remove_handwritten_from_csv

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)


# remove the handwritten image caption pairs from train_deepl.csv 
# for both original and augmented 
# TODO : Add option to do the same pipeline for validation set as well
# it will be done here 

if __name__=="__main__":
    train_df1 = remove_handwritten_from_csv('./Datasets/original/train_deepl.csv','./handwritten_images')
    train_df1.to_csv('./Datasets/original/train_deepl.csv')
    print(f'New length for train dataset  : {len(train_df1)}\n')

    train_df2 = remove_handwritten_from_csv('./Datasets/augmented/train_deepl.csv','./handwritten_images')
    train_df2.to_csv('./Datasets/augmented/train_deepl.csv')
    print(f'New length for train dataset  : {len(train_df2)}\n')
