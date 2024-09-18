import os
import sys

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)
import pandas as pd
import os
import json
from datasets import load_dataset
from torch.utils.data import Dataset
import shutil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from torchvision.transforms import transforms as T
from PIL import Image
import nlpaug.augmenter.char as nacs
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word.context_word_embs as nawcwe
import nlpaug.augmenter.word.word_embs as nawwe
import nlpaug.augmenter.word.spelling as naws
from fix_images_paths import DatasetsPreprocessor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from config_files import *
from fix_images_paths import *
import torch
import clip
from utils import *
from creating_datasets import original_path, augmented_path
import time
from paths_config import *
import glob
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hugda, DatasetDict
import time
from PIL import Image
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word.context_word_embs as nawcwe
import nlpaug.augmenter.word.word_embs as nawwe
import nlpaug.augmenter.word.spelling as naws
from config_files import *
from fix_images_paths import *
from creating_datasets import *



# calling the classes on runtime 
# we will use different languages fo backtranslation 
fr_aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-fr', 
    to_model_name='Helsinki-NLP/opus-mt-fr-en', 
)

it_aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-it', 
    to_model_name='Helsinki-NLP/opus-mt-it-en', 
)
es_aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-es', 
    to_model_name='Helsinki-NLP/opus-mt-es-en', 
)
pt_aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-de', 
    to_model_name='Helsinki-NLP/opus-mt-de-en', 
)




# just a placeholder class to return the caption unchanged 
class Nobacktranslation:
    def __init__(self):
        pass 

    def augment(self,x):
        return x 




if __name__=="__main__":
    # for augmented only create the postprocessed csv
    # comment out if csv is already cleaned and preprocessed 
    #dp = DatasetsPreprocessor("train_deepl.csv")
    #dp.fix_image_paths(path=augmented_path)

    # for train dataset only 

    # set seed 
    torch.manual_seed(0)

    no_aug = Nobacktranslation()

    # list of backtranslation objects for the 4 languages 
    # the first augmented caption will be the true caption 
    # so for each pair we will have 1 true caption, 4 augmented captions
    augs = [no_aug,fr_aug,it_aug,es_aug,pt_aug]



    # before applying the augmentation strategy to generate the new augmented dataset 
    # we will split the train dataset into train and val so we leave the val dataset untouched 
    total_df = pd.read_csv('/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv')
    train_df, val_df = train_test_split(total_df, test_size=0.1, random_state=0,stratify=total_df['category'])
    train_df.reset_index(inplace=True,drop=True)
    train_df.drop(columns=['Unnamed: 0.1','Unnamed: 0'],inplace=True,errors='ignore')
    val_df.drop(columns=['Unnamed: 0.1','Unnamed: 0'],inplace=True,errors='ignore')
    val_df.reset_index(drop=True,inplace=True) 
    # now write the train data, val data to new csv 
    train_df.to_csv('/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv')
    val_df.to_csv('/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/val_deepl_augmented.csv')
    print("original columns are : ",train_df.columns)


    augmenter = DatasetAugmentGenerator(
                    augs,
                    savepath ='/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/trainimages',
                    final_path2 = '/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv',
                    starting_path='/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv',
                    img_repf=5,
                    cap_repf=5)
    # we overwrite the train_deepl.csv 


    # add postprocessing step to clean the new augmented : train_deepl.csv
    # TODO : ADD A POSTPROCESSING STEP TO DROP ANY WORDS REPEATED N TIMES CONSECUTIVELY
    # IN THIS CASE WE KEEP ONLY ONE APPEARANCE OF THE WORD
    # APPLY IT AND OVERWRITE THE train_deepl.csv again


    


    # create the augmented dataset 
    start_time = time.time()
    augmenter(augs)
    end_time=time.time()-start_time
    print(f'Time taken for augmentation: {str(end_time)}')


    

if __name__=="__main__" :
    # APPLY POSTPROCESSING HERE TO CLEAN THE TRAINING DATASET 
    # remove rows with meaning less duplicate words (errors from backtranslation)
    dup_cleaner = DuplicateCleaner('/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv')
    clean_augmented_train = dup_cleaner()

    # save the train csv 
    clean_augmented_train.to_csv('/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv')
    print('Creating the Image Caption pairs')
    # add the image caption pairs now and create the jsonl
    # of the augmented dataset 
    # normally we better do this in another file so we don't have a read after write
    # TODO : don't we need the preprocessing part ??  to drop nulls etc / apply regex replacements 
    dp1 = DatasetsPreprocessor("train_deepl.csv")
    dp1.copy_images(path=augmented_path, add_category=True,images_exist=True)


    # TODO : Check if we also need to create the image caption pairs for the test/val datasets 
    # TODO : For the augmentated dataset training we still can use the origina test/val so we probably should no create the image/caption pairs again
    # for those we create the simple original image caption pairs 
    
    # read the new split validation dataset (from the split of the training dataset and create the images/jsonl)
    # TODO : don't we need the preprocessing part ??  to drop nulls etc / apply regex replacements 
    dp2 = DatasetsPreprocessor("val_deepl.csv")
    #dp2.fix_image_paths(path=augmented_path)
    #dp2.copy_images(path=augmented_path, add_category=True)
    dp2.copy_images(path=augmented_path, add_category=True)


    # here use the normal preprocessing for the test set 
    dp3 = DatasetsPreprocessor("test_deepl.csv")
    dp3.fix_image_paths(path=augmented_path)
    dp3.copy_images(path=augmented_path, add_category=True)


    
    print("=" * 100)
    print("Augmented Dataset's csv files were created")
    print("=" * 100)
    



