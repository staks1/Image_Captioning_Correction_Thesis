import sys
import os

# Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to sys.path if it's not already there
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)


import pandas as pd
import json
from datasets import load_dataset
from torch.utils.data import Dataset
import shutil
from nltk.tokenize import word_tokenize
import numpy as np
from torchvision.transforms import transforms as T
from PIL import Image
from fix_images_paths import DatasetsPreprocessor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from config_files import *
import torch
import clip
from utils import *
import time
import torch.nn.functional as F
from calculate_metrics import calculate_metrics
import nltk



def top_5_captions(test_data_images):
    _, axs = plt.subplots(1, len(test_data_images), figsize=(12, 12))
    axs = axs.flatten()
    results = {}

    # Preload all caption embeddings to NumPy
    # cap_emb_np = cap_emb.cpu().numpy()  # shape: [num_captions, feature_dim]

    # Extract all captions
    captions = [item["text"] for item in test_data]

    for i, (x, ax) in enumerate(zip(test_data_images, axs)):
        # Query image
        img_query = test_data[x]["image"]

        # Preprocess and encode image
        img_query_tensor = preprocess(img_query).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img_query_tensor)

        # Compute similarity (dot product)
        similarities = image_features @ cap_emb.T  # shape: [1, num_captions]

        # Get top-5 indices and values using NumPy
        k = 5
        topk_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_indexes_list = topk_indices.tolist()

        # Map indices to captions
        top_k_captions_list = [
            [captions[idx] for idx in row] for row in top_k_indexes_list
        ]
        # captions = [test_data[i]['text'] for i in range(len(test_data))]
        # Store and visualize
        results[x] = top_k_captions_list
        ax.imshow(img_query)
        ax.axis("off")
    return results


if __name__ == "__main__":
    print("Available models:\n", clip.available_models())
    print("Using ViT-L-14.pt\n")
    # load model to create the embeddings
    # if the mode is saved locally in the
    # Models/zero-shot directory then this will be used
    # otherwise it is downloaded from clip repo
    try:
        model, preprocess = from_model(path_to_weights_zo, device="cpu")
    except:
        device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)

    # create the dataset from huggingface format
    # compatible with torch
    # creating a small sample of the dataset (100 samples)
    dp3 = DatasetsPreprocessor("test_deepl.csv")
    dp3.fix_image_paths(path="./Datasets/original/")
    dp3.copy_images(path="./Datasets/original/", add_category=True)
    # create the embeddings for the small dataset

    dp = DatasetsPreprocessor("test_deepl.csv")
    test_data = dp.create_Dataset(path="./Datasets/original/")

    test_data_images = [0, 1, 2, 3, 4]

    print("Calculating embeddings\n")
    # calculate embeddings for the 100 samples
    model.to("cpu")

    clip_dataset = ClipDataset(
        test_data,
        preprocess=preprocess,
        tokenize=clip.tokenize,
        label_encoder=None,
        classification=False,
    )

    # pick device
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # the order is useful to later compare embeddings
    encode_dataloader = DataLoader(clip_dataset, batch_size=5, shuffle=False)

    # time to calculate embeddings
    start_time = time.time()
    img_emb, cap_emb = clip_embed(
        encode_dataloader, model, 768, device, return_numpy=True
    )
    print(f"Embeddings have shape {img_emb.shape}\n")

    end_time = time.time() - start_time
    print("saving embeddings\n")
    save_embeddings(
        data_model_code="zo",
        split_path="test_deepl.csv",
        img_emb=img_emb,
        caption_emb=cap_emb,
    )
    print("Calulating metrics\n")
    calculate_metrics(
        data_model_code="zo",
        split_path="test_deepl.csv",
        dataset_path="./Datasets/original/",
        top_k=3,
        is_numpy=True,
    )

    test_data_images = [x for x in range(len(test_data))]

    print("The top 5-matching captions for each image are as follows:\n")
    results = top_5_captions(test_data_images)
    for i, x in enumerate(results):
        print(f"IMAGE ({x}) CAPTIONS: (TOP-5)\n")
        print(results[x], "\n\n")
