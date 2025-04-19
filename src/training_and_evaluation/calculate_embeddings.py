# This is the beginning of the pipeline to evaluate zeroshot clip on my dataset and
# also generate the instructions to correct the problems

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


# load the original training dataset
# TODO : we will also potentially change this with the augmented dataset to calculate results on this as well


# split_path can be "train_deepl.csv","val_deepl.csv" or "test_deepl.csv"
# dataset_path can be original_path or augmented_path
def calculate_embeddings(split_path, dataset_path, data_model_code, path_to_weights):

    torch.manual_seed(0)  # For reproducibility
    dp = DatasetsPreprocessor(split_path)
    train_data = dp.create_Dataset(dataset_path)

    print(f"CLIP AVAILABLE MODELS:{clip.available_models()}\n")
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # we will use cuda by default unless we find any errors
    # model_name = "ViT-L/14"

    print(f"Using device : {device}\n")
    model, preprocess = from_model(path_to_weights, device=device)
    # can give model_name or path to saved model weights

    # creating the Clip compatible dataset to calculate the embeddings

    clip_dataset = ClipDataset(
        train_data,
        preprocess=preprocess,
        tokenize=clip.tokenize,
        label_encoder=None,
        classification=False,
    )

    # creating the encoding dataloader to create the batches for the encoding
    # since we only build the embeddings and do no training here we use bs = 1, and no shuffle
    # since the order is useful to later compare embeddings
    encode_dataloader = DataLoader(clip_dataset, batch_size=5, shuffle=False)

    # create the embeddings
    start_time = time.time()
    print("Calculating Embeddings \n")
    img_emb, cap_emb = clip_embed(encode_dataloader, model, 768, device)
    time_for_embeddings = time.time() - start_time

    # save the embeddings to arrays (not lists)
    # maybe also save numpy/pickle instead
    save_embeddings(data_model_code, split_path, img_emb, cap_emb)

    print(f"Time taken to create embeddings : {str(time_for_embeddings)}\n")
    # save also the time for zero shot for each dataset
    time_to_embed(time_for_embeddings, data_model_code, split_path)


# run for test dataset and the original dataset and zero clip model
# in main
# TODO : Do for the other splits as well
if __name__ == "__main__":
    calculate_embeddings("test_deepl.csv", original_path, "zo", path_to_weights_zo)
