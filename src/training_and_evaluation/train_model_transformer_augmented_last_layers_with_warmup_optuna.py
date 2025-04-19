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
import nlpaug.augmenter.word.context_word_embs as nawcpwe
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
from transformers import get_scheduler
import optuna

# set torch seed
torch.manual_seed(0)
np.random.seed(0)

# define checkpoints for continuing training in case of disconnection or error
# Define paths for checkpointing
checkpoint_path = path_to_weights_tla + "checkpoint_mod.pt"
dataset_path = augmented_path
split_path = "train_deepl.csv"


dp = DatasetsPreprocessor(split_path)
train_data = dp.create_Dataset(dataset_path)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
print(f"Using device : {device}\n")

# create the validation dataset
dp2 = DatasetsPreprocessor("val_deepl.csv")
val_data = dp2.create_Dataset(dataset_path)


# static hyperparameters
# ------------------------#
# batch_size = 5
num_epochs = 10
# early stopping
patience = 8  # Number of epochs to wait for improvement
patience_counter = 0  # Counter for early stopping

# contstruct the dataloaders for training/validation set


val_losses = []
train_losses = []
start_time = time.time()

best_global_loss = float("inf")
best_global_trial_info = None


# function to minimize
def objective(trial, batch_size, best_global_loss):

    # define model here
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # suggest batch size
    # batch_size = trial.suggest_categorical('batch_size', [5, 10])
    # set the batch size as usr attribute so we can retrieve the best batch size later
    trial.set_user_attr("batch_size", batch_size)

    clip_train_data = ClipDataset(
        dataset=train_data, preprocess=preprocess, tokenize=clip.tokenize
    )
    train_dataloader = DataLoader(clip_train_data, batch_size=batch_size, shuffle=True)

    clip_val_data = ClipDataset(
        dataset=val_data, preprocess=preprocess, tokenize=clip.tokenize
    )
    val_dataloader = DataLoader(clip_val_data, batch_size=batch_size, shuffle=False)
    # final total steps of training
    total_steps = len(train_dataloader) * num_epochs

    # optuna trial hyperparams
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    num_layers_to_finetune = trial.suggest_int("num_layers_to_finetune", 6, 10)
    model = freeze_last_k_transformer_layers(model, num_layers_to_finetune)
    print_only_trainable(model)
    optimizer = pick_hyperaparameters_last_k_transformer_layers_custom(
        model, last_k=num_layers_to_finetune, lr=learning_rate, wd=weight_decay
    )
    warmup_steps = trial.suggest_loguniform("warmup_steps", 0.01, 0.2)

    # add warmup scheduler
    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine"])

    scheduler = get_scheduler(
        name=scheduler_type,  # You can use "cosine" for cosine annealing
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps * total_steps),
        num_training_steps=total_steps,
    )

    ############# TRAINING LOOP ################
    # set up plot loss lists
    val_losses = []
    train_losses = []

    start_time = time.time()
    # set up for training
    model.train()
    print("Starting training\n")

    best_loss = float("inf")  # very high loss

    for epoch in range(num_epochs):
        running_loss = 0.0

        # training
        for images, captions in train_dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # do we need to define the encode_image, encode_text method or not ?
            # TODO : SOS check if we need to define our own encoding like in zero_shot
            image_features = model.encode_image(images)
            text_features = model.encode_text(captions)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            logits_per_image, logits_per_text = model(images, captions)

            # Labels for contrastive loss
            batch_size = images.size(0)
            labels = torch.arange(batch_size, device=device)

            # Contrastive loss
            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            # TODO  : MAYBE USE THE NORMAL AVERAGE LOSS
            # set higher image to text loss than text to image
            loss = 0.6 * loss_img + 0.4 * loss_txt

            # Backward pass
            loss.backward()

            # to prevent gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # gradient descent
            optimizer.step()

            # sceduler update
            scheduler.step()

            # Update running loss
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # validation
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_images, val_captions in val_dataloader:
                val_images = val_images.to(device)
                val_captions = val_captions.to(device)

                # Forward pass
                val_image_features = model.encode_image(val_images)
                val_text_features = model.encode_text(val_captions)

                # Normalize features
                val_image_features = val_image_features / val_image_features.norm(
                    dim=-1, keepdim=True
                )
                val_text_features = val_text_features / val_text_features.norm(
                    dim=-1, keepdim=True
                )

                # Compute similarity
                val_logits_per_image, val_logits_per_text = model(
                    val_images, val_captions
                )

                # Labels for contrastive loss
                val_batch_size = val_images.size(0)
                val_labels = torch.arange(val_batch_size, device=device)

                # Contrastive loss
                val_loss_img = torch.nn.functional.cross_entropy(
                    val_logits_per_image, val_labels
                )
                val_loss_txt = torch.nn.functional.cross_entropy(
                    val_logits_per_text, val_labels
                )
                # AGAIN HERE USE HUGHER WEIGHT FOR IMAGE-TO-TEXT LOSS
                val_loss = 0.6 * val_loss_img + 0.4 * val_loss_txt

                # Update running loss
                val_running_loss += val_loss.item() * val_images.size(0)

        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        val_losses.append(val_epoch_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}"
        )

        # Save the best model based on validation loss
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            model_path = os.path.join(
                path_to_weights_tla, f"best_model_trial_{trial.number}.pt"
            )
            trial.set_user_attr("best_model_path", model_path)
            torch.save(model, model_path)
            print(f"Model saved with Validation Loss: {best_loss:.4f}")
            patience_counter = 0  # Reset the counter if we get a better model
        else:
            patience_counter += 1

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # we need to also set the global best loss for optuna if the model is the current best
        if best_loss < best_global_loss:
            print(f"New best global loss found: {best_loss:.4f}")
            best_global_loss = best_loss

            model_path = trial.user_attrs[
                "best_model_path"
            ]  # Path for this trial's model
            best_global_trial_info = {
                "trial_number": trial.number,
                "best_loss": best_loss,
                "hyperparameters": trial.params,
                "model_path": model_path,
            }
            print("best params are ", trial.params)

            # store the results in json
            with open(
                os.path.join(path_to_weights_tla, "best_global_trial_info.txt"), "a+"
            ) as f:
                json.dump(best_global_trial_info, f, indent=4)

        # Check early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            return best_loss
            # break
    return best_loss


# ------------ MAIN RUN ----------------------------#
# call optuna study and try to minimize the function
study = optuna.create_study(direction="minimize")  # We are minimizing validation loss


# test both batch sizes
batch_sizes = [128]
for batch_size in batch_sizes:
    study.optimize(
        lambda trial: objective(trial, batch_size, best_global_loss), n_trials=10
    )


end_time = time.time() - start_time
print("total training time is : ", end_time)

# write time to weights folder
with open(os.path.join(path_to_weights_tla, "training_time.txt"), "a+") as f:
    f.write(f"Training time : {str(end_time)}")


# save the loss function plot
plot_losses(train_losses, val_losses, path_to_plot_tla + "train_val_loss_plot.png")
# ------------------------#
# train model for now !
# train_clip_model(model,train_dataloader, optimizer,device,path_to_weights_plo, num_epochs=10)
