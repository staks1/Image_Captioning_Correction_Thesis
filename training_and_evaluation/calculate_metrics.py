
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
from creating_datasets import original_path,augmented_path
import time
import torch.nn.functional as F
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from nltk.translate.meteor_score import single_meteor_score
#from pyciderevalcap.cider.cider import Cider


def calculate_metrics(data_model_code,split_path,dataset_path,top_k):
        torch.manual_seed(0)  # For reproducibility
        # load the embeddings 
        image_embeddings,caption_embeddings = load_embeddings(data_model_code,split_path)

        # calculate similarities 
        similarities = image_embeddings @ torch.t(caption_embeddings)

        topk_values,topk_indices = torch.topk(similarities,k=top_k,dim=1)
        top_k_indexes_list = topk_indices.tolist()

        dp = DatasetsPreprocessor(split_path)
        evaluation_data = dp.create_Dataset(dataset_path)

        captions = [evaluation_data[i]['text'] for i in range(len(evaluation_data))]
        top_k_captions_list = [[captions[caps] for caps in im] for im in top_k_indexes_list]

        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smooth = SmoothingFunction().method1
        rouge_scores=[]
        bleu_scores=[]

        for i, top_k_captions in enumerate(top_k_captions_list):
            # Get the reference caption(s) for the current image
            reference_caption = evaluation_data[i]['text']


            # Compute ROUGE scores
            rouge_score_per_image = [rouge_scorer_instance.score(reference_caption, predicted_caption) for predicted_caption in top_k_captions]
            rouge_scores.append(rouge_score_per_image)

            # Compute BLEU scores
            bleu_score_per_image = [sentence_bleu([reference_caption.split()], predicted_caption.split(), smoothing_function=smooth) for predicted_caption in top_k_captions]
            bleu_scores.append(bleu_score_per_image)

        cumulative_rouge = {'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                    'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                    'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}}


        # Average  Rouge scores 
        num_images = len(rouge_scores)
        for rouge_score_per_image in rouge_scores:
            for score in rouge_score_per_image:
                for metric in cumulative_rouge.keys():
                    cumulative_rouge[metric]['precision'] += score[metric].precision
                    cumulative_rouge[metric]['recall'] += score[metric].recall
                    cumulative_rouge[metric]['fmeasure'] += score[metric].fmeasure

        # average rouge 
        average_rouge = {metric: {key: val / (num_images * top_k) for key, val in scores.items()} for metric, scores in cumulative_rouge.items()}
        # Compute average BLEU score
        average_bleu = sum([sum(scores) for scores in bleu_scores]) / (num_images * top_k)

        save_results(data_model_code,split_path,average_rouge,average_bleu)

        print(f'The average ROUGE score for the dataset is : {average_rouge}')
        print(f'The average BLEU score for the dataset is : {average_bleu}')






# run with zo,test for zero clip and original dataset,test
if __name__ == "__main__":
    # ==== Run for each Model ====
    #calculate_metrics("zo","test_deepl.csv",original_path,5)
    #calculate_metrics("plo","test_deepl.csv",original_path,5)
    calculate_metrics("tlo","test_deepl.csv",original_path,5)