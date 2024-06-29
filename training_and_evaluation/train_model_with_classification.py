
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


# set torch and numpy seed 
np.random.seed(0)             
torch.manual_seed(0)

# define checkpoints for continuing training in case of disconnection or error 
# Define paths for checkpointing
checkpoint_path = path_to_weights_clo + 'checkpoint_mod.pt'

# give batch size,split_path,original_path
#---------------------------------
batch_size = 64
split_path = "train_deepl.csv"
dataset_path = original_path
num_epochs = 30
# early stopping 
patience = 5  # Number of epochs to wait for improvement
patience_counter = 0  # Counter for early stopping
#---------------------------------

dp = DatasetsPreprocessor(split_path)
train_data = dp.create_Dataset(dataset_path)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f'Using device : {device}\n')

# create dataframe from train_dataset 
train_df = pd.DataFrame(train_data) # convert to dataframe
# split into train/val , with same class balance as training set 
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=0,stratify=train_df['label'])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True) 

# Convert the DataFrames back to Hugging Face Datasets
train_dataset = df_to_dataset(train_df)
val_dataset = df_to_dataset(val_df)

# set up classes for the classification 
classes = [category[0] for category in train_df[['label']].value_counts().index]
num_classes = len(classes)
print(num_classes)
print(classes)


################# MODEL ###########################
model,preprocess = clip.load('ViT-L/14', device=device, jit=False)
# for now load the already (until a point trained model to continue training)
#model = torch.load('./Models/clip_trained_projection_layers/original/weights/ViT-L-14.pt')

model = freeze_clip_projection_layers(model,unfreeze_transformer_last=False)


# also unfreeze the final normalization layers 
for p in model.ln_final.parameters():
    p.requires_grad = True 
print_only_trainable(model)



# creating the classifier on top model 
clip_projection_dim = model.ln_final._parameters['weight'].shape[0]
embedding_dim = clip_projection_dim
model_with_classifier = CLIPWithClassifier(model,num_classes,embedding_dim)
print_only_trainable(model_with_classifier)



# initialize label encoder and fit it to train dataset , create dataloaders 
label_encoder = init_label_encoder(train_dataset)
clip_train_data  = ClipDataset(dataset=train_dataset,preprocess=preprocess,tokenize=clip.tokenize,classification=True,label_encoder=label_encoder)
train_dataloader = DataLoader(clip_train_data,batch_size=batch_size,shuffle=True)
clip_val_data = ClipDataset(dataset=val_dataset,preprocess=preprocess,tokenize=clip.tokenize,classification=True,label_encoder=label_encoder)
val_dataloader = DataLoader(clip_val_data,batch_size=batch_size,shuffle=False)


# TODO : we will also try with optuna 
optimizer = pick_hyperaparameters(model,unfreeze_transformer_last=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model_with_classifier.fc.parameters()},
    {'params': model_with_classifier.clip_model.visual.proj, 'lr': 1e-5,'weight_decay': 1e-4},
    {'params': model_with_classifier.clip_model.text_projection, 'lr': 1e-5,'weight_decay': 1e-4}
], lr=0.0001)




# set up plot loss lists 
val_losses = []
train_losses = []

############# TRAINING LOOP ################


start_time = time.time()
# set up for training 
model.train()
print("Starting training\n")

best_loss = float('inf') # very high loss 

for epoch in range(num_epochs):
    running_loss = 0.0

    # training 
    for images, captions,categories in train_dataloader:
        images = images.to(device)
        captions = captions.to(device)
        categories = categories.to(device)


        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute similarity
        outputs = model(images, captions)
        print(outputs)
        loss = criterion(outputs, categories)

        # Backward pass
        loss.backward()

        # to prevent gradient exploding 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

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
        for val_images, val_captions,val_categories in val_dataloader:
            val_images = val_images.to(device)
            val_captions = val_captions.to(device)
            val_categories = val_categories.to(device)


            # Compute similarity
            val_outputs = model(val_images, val_captions)
            val_loss = criterion(val_outputs,val_categories)
           
            # Update running loss
            val_running_loss += val_loss.item() * val_images.size(0)

    val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
    val_losses.append(val_epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}")

    # Save the best model based on validation loss
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        torch.save(model, path_to_weights_clo + "ViT-L-14.pt")
        print(f"Model saved with Validation Loss: {best_loss:.4f}")
        patience_counter = 0  # Reset the counter if we get a better model
    else:
        patience_counter += 1

    # save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Check early stopping condition
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break


end_time = time.time() - start_time
# write time tp weights folder 
with open(os.path.join(path_to_weights_clo,"training_time.txt"),"a+") as f:
    f.write(f"Training time : {str(end_time)}")

# save the loss function plot 
plot_losses(train_losses, val_losses, path_to_plot_clo + "train_val_loss_plot.png")  

#------------------------#
# train model for now !
#train_clip_model(model,train_dataloader, optimizer,device,path_to_weights_plo, num_epochs=10)
