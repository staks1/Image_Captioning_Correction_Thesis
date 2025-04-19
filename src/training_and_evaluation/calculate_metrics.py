import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
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
import torch.nn.functional as F
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_metrics(data_model_code, split_path, dataset_path, top_k, is_numpy=False):
    torch.manual_seed(0)  # For reproducibility
    # load the embeddings
    image_embeddings, caption_embeddings = load_embeddings(
        data_model_code, split_path, is_numpy=is_numpy
    )

    # calculate similarities
    similarities = image_embeddings @ torch.t(caption_embeddings)

    # softmax to turn to probabilities
    probabilities = F.softmax(similarities / 0.07, dim=1)

    topk_values, topk_indices = torch.topk(probabilities, k=top_k, dim=1)
    top_k_indexes_list = topk_indices.tolist()

    dp = DatasetsPreprocessor(split_path)
    evaluation_data = dp.create_Dataset(dataset_path)

    captions = [evaluation_data[i]["text"] for i in range(len(evaluation_data))]
    top_k_captions_list = [[captions[caps] for caps in im] for im in top_k_indexes_list]

    print("len of test set is", len(top_k_captions_list))
    rouge_scorer_instance = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    smooth = SmoothingFunction().method1
    rouge_scores = []
    bleu_scores = []
    num_images = len(top_k_captions_list)

    # here write
    cumulative_rouge = {
        "rouge1": {"precision": 0, "recall": 0},
        "rouge2": {"precision": 0, "recall": 0},
        "rougeL": {"precision": 0, "recall": 0},
    }

    # Iterate over top-k captions for each image
    for i, top_k_captions in enumerate(top_k_captions_list):
        reference_caption = evaluation_data[i]["text"]

        prec_per_ref = {
            "rouge1": {"precision": 0},
            "rouge2": {"precision": 0},
            "rougeL": {"precision": 0},
        }
        rec_per_ref = {
            "rouge1": {"recall": 0},
            "rouge2": {"recall": 0},
            "rougeL": {"recall": 0},
        }

        # for each reference caption
        for candidate_caption in top_k_captions:
            # Compute ROUGE scores for this candidate caption
            score = rouge_scorer_instance.score(reference_caption, candidate_caption)

            # add recall
            rec_per_ref["rouge1"]["recall"] += score["rouge1"].recall
            rec_per_ref["rouge2"]["recall"] += score["rouge2"].recall
            rec_per_ref["rougeL"]["recall"] += score["rougeL"].recall

            # add precision
            prec_per_ref["rouge1"]["precision"] += score["rouge1"].precision
            prec_per_ref["rouge2"]["precision"] += score["rouge2"].precision
            prec_per_ref["rougeL"]["precision"] += score["rougeL"].precision

        # add to the cumulative
        cumulative_rouge["rouge1"]["precision"] += prec_per_ref["rouge1"]["precision"]
        cumulative_rouge["rouge2"]["precision"] += prec_per_ref["rouge2"]["precision"]
        cumulative_rouge["rougeL"]["precision"] += prec_per_ref["rougeL"]["precision"]

        cumulative_rouge["rouge1"]["recall"] += rec_per_ref["rouge1"]["recall"]
        cumulative_rouge["rouge2"]["recall"] += rec_per_ref["rouge2"]["recall"]
        cumulative_rouge["rougeL"]["recall"] += rec_per_ref["rougeL"]["recall"]

    # Normalize by the total number of scores considered
    average_rouge = {
        metric: {
            "precision": cumulative_rouge[metric]["precision"] / (num_images * top_k),
            "recall": cumulative_rouge[metric]["recall"] / (num_images * top_k),
        }
        for metric in ["rouge1", "rouge2", "rougeL"]
    }

    save_results(data_model_code, split_path, average_rouge, "")


# run with zo,test for zero clip and original dataset,test
if __name__ == "__main__":
    # ==== Run for each Model ====
    calculate_metrics("zo", "test_deepl.csv", original_path, 1)
    calculate_metrics("zo", "test_deepl.csv", original_path, 5)
    calculate_metrics("zo", "test_deepl.csv", original_path, 10)
