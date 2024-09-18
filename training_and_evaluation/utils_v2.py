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
import torch.nn.functional as F
from paths_config import *
import glob 
import torch.optim as optim
from datasets import load_dataset, Dataset as hugda, DatasetDict
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import shutil
import glob 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# clip dataset class 
class ClipDataset(Dataset):
    def __init__(
        self,
        dataset,
        preprocess,
        tokenize=clip.tokenize,
        label_encoder=None,
        classification=False,
    ):

        self.dataset = dataset
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.classification = classification
        #if self.classification:
        #    self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        images = item["image"]
        captions = item["text"]
        #TODO : delete next line if problem arises
        labels = item["label"]
        images = self.preprocess(images)
        # SOS pass truncate = True to prevent crashing the model with very large captions
        # remove truncate if it does not work
        captions = self.tokenize(captions,truncate=True)
        captions = captions.squeeze()

        # encode the string labels(category) into integers
        if self.classification:
            # TODO : uncomment next line if it does not work
            #labels = self.label_encoder.transform([item["label"]])[0]
            #print(labels)
            return images, captions, labels
        else:
            return images, captions

# classifier model to add on top of clip model 
class CLIPWithClassifier(nn.Module):
    def __init__(self, clip_model, num_categories,projection_dim):
        super(CLIPWithClassifier, self).__init__()
        self.clip_model = clip_model

        # defining the linear classifier on top 
        self.fc = nn.Linear(projection_dim, num_categories)  
        
        # freeze the other parameters 
        for par in self.clip_model.parameters():
            par.requires_grad = False
            
        self.clip_model.visual.proj.requires_grad = True
        self.clip_model.text_projection.requires_grad= True
        for p in self.clip_model.ln_final.parameters():
                p.requires_grad = True 
        

    # forward pass 
    def forward(self, images, captions):
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(captions)
        

        # TODO : How do we combine the 2 embeddings (image + caption embeddings)
        # simple addition ? 
        # what else can we do ? 
        combined_features = image_features + text_features  # Simple way to combine features
        print(combined_features.shape)
        output = self.fc(combined_features)
        #print(output.shape)
        return output


# fit label encoder for classification
def init_label_encoder(dataset):
    label_encoder = LabelEncoder()
    label_encoder.fit([item['label'] for item in dataset])
    return label_encoder


# function load each different model to calculate the embeddings 
# change device to 'cuda' if needed , to 'cpu' if needed (for classification model)
def from_model(path_to_model,device='cuda'):
    # for all trained/untrained models with clip compatible architecure (no layers removed or added)
    if(path_to_model==path_to_weights_zo):
        # only for this load the initial pretrained clip 
        model, preprocess = clip.load(path_to_weights_zo + 'ViT-L-14.pt', device=device, jit=False)
    elif(path_to_model == path_to_weights_plo or path_to_model==path_to_weights_pla \
       or path_to_model == path_to_weights_tlo or path_to_model==path_to_weights_tla):
       # TODO : in order for this to work --> the name of the saved weights must be one of the available_models() of clip!!!
        temp_path = glob.glob(os.path.join(path_to_model ,"*.pt"))[0]
        model = torch.load(temp_path,map_location=torch.device(device))
        _,preprocess = clip.load(path_to_weights_zo + 'ViT-L-14.pt', device=device, jit=False)
        print(f"Using model from : {temp_path}\n")
    else :
        # for extended model with classification layers on top 
        _, preprocess = clip.load(path_to_weights_zo +'ViT-L-14.pt', device=device, jit=False)
        temp_path = glob.glob(os.path.join(path_to_model ,"*.pt"))[0]
        extended_model = torch.load(temp_path,map_location=torch.device(device)) # get the first model in the dir
        # this is needed only for classification training
        # for inference we need again the whole model 
        model = extended_model.clip_model # extract only the trained clip model 
    
    return model,preprocess
    
# TODO : FIX from_model function
# during training we need device=cpu,model = extended_model.clip_model
# during inference we need device=cuda,model = extended_model
# for all the other models we use cuda








# function to save embeddings 
# it is based on the different models and datasets combination
# TODO : Maybe needs refactoring ,also add for other symbols--> models/datasets
# this takes the path and the image and caption embeddings and saves them in the path
def save_embeddings(data_model_code,split_path, img_emb, caption_emb):
    if(data_model_code=='zo' and split_path=='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_zo_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_zo_test, "caption_embeddings.pt"))
        print("Embeddings saved. \n")
    # add for next models 
    elif(data_model_code=='plo' and split_path =='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_plo_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_plo_test, "caption_embeddings.pt"))
    elif(data_model_code=='pla' and split_path =='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_pla_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_pla_test, "caption_embeddings.pt"))
    elif(data_model_code=='tlo' and split_path =='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_tlo_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_tlo_test, "caption_embeddings.pt"))
    elif(data_model_code=='tla' and split_path =='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_tla_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_tla_test, "caption_embeddings.pt"))
    elif(data_model_code=='clo' and split_path =='test_deepl.csv'):
        torch.save(img_emb,  os.path.join(path_to_embeddings_clo_test,"image_embeddings.pt"))
        torch.save(caption_emb, os.path.join(path_to_embeddings_clo_test, "caption_embeddings.pt"))



# function to load the correct embeddings 
def load_embeddings(data_model_code,split_path):
    if(data_model_code=='zo' and split_path=='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_zo_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_zo_test, "caption_embeddings.pt"))

    # add for next models 
    elif(data_model_code=='plo' and split_path =='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_plo_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_plo_test, "caption_embeddings.pt"))

    # add for next models 
    elif(data_model_code=='pla' and split_path =='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_pla_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_pla_test, "caption_embeddings.pt"))
    # add for next models 
    elif(data_model_code=='tlo' and split_path =='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_tlo_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_tlo_test, "caption_embeddings.pt"))
    elif(data_model_code=='tla' and split_path =='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_tla_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_tla_test, "caption_embeddings.pt"))
    elif(data_model_code=='clo' and split_path =='test_deepl.csv'):
        img_emb = torch.load(os.path.join(path_to_embeddings_clo_test,"image_embeddings.pt"))
        caption_emb = torch.load(os.path.join(path_to_embeddings_clo_test, "caption_embeddings.pt")) 
    
    return img_emb,caption_emb



# add function to save the resulting metrics    
def save_results(data_model_code,split_path,rouge_metric,bleu_metric):
    if(data_model_code =='zo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_zo_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")
    elif(data_model_code =='plo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_plo_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")
    elif(data_model_code =='pla' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_pla_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")
    elif(data_model_code =='tlo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_tlo_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")
    elif(data_model_code =='tla' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_tla_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")
    elif(data_model_code =='clo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_results_clo_test ,"results.txt"),"a+") as f:
            f.write(f'ROUGE : {str(rouge_metric)}\n')
            f.write(f'BLEU :  {str(bleu_metric)}\n')
        print("Results written. \n")

    
    # add code for the other models 
    # elif ...




# time embeddings calculation based on model
# TODO : Check which embeddings we will save
# guessing only the test_embeddings 
def time_to_embed(time_taken,data_model_code,split_path):
    if(data_model_code=='zo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_embeddings_zo_test,"embeddings_time.txt"),"a+") as f:
            f.write(str(time_taken))
        print('Time for Embeddings calculation written. \n')
    
    elif(data_model_code =='plo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_embeddings_plo_test,"embeddings_time.txt"),"a+") as f:
            f.write(str(time_taken))
        print('Time for Embeddings calculation written. \n')

    elif(data_model_code =='tlo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_embeddings_tlo_test,"embeddings_time.txt"),"a+") as f:
            f.write(str(time_taken))
        print('Time for Embeddings calculation written. \n')
    elif(data_model_code =='clo' and split_path=='test_deepl.csv'):
        with open(os.path.join(path_to_embeddings_clo_test,"embeddings_time.txt"),"a+") as f:
            f.write(str(time_taken))
        print('Time for Embeddings calculation written. \n')
    # add for the other models 
    #elif(data_model_code=='plo'):
    #elif(data_model_code=='pla'):
    #elif(data_model_code=='tlo'):
    #elif(data_model_code=='tla'):
    #elif(data_model_code=='clo'):
    #elif(data_model_code=='cla'):   
    #elif(data_model_code=='cltlo'):
    #elif(data_model_code=='cltlo'):  



# torch cpu/gpu encode image
def encode_image(image, model, device="cuda", return_numpy=False):
    # model.to(device)
    image = image.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    if return_numpy:
        return image_features.cpu().numpy()
    else:
        return image_features


# torch cpu/gpu encode text
def encode_text(text, model, device="cuda", return_numpy=False):
    text = text.to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    if return_numpy:
        return text_features.cpu().numpy()
    else:
        return text_features




# define second versions of embed to do it with torch ,and gpu
def clip_embed(data_loader, model, embedding_dim, device="cuda", return_numpy=False):

    num_samples = len(data_loader.dataset)
    image_embeddings = torch.zeros((num_samples, embedding_dim), device=device)
    text_embeddings = torch.zeros((num_samples, embedding_dim), device=device)

    idx = 0
    for i, x in enumerate(data_loader):
        images = x[0].to(device)
        texts = x[1].to(device)

        with torch.no_grad():
            image_embedding = encode_image(images, model, device)
            text_embedding = encode_text(texts, model, device)

        batch_size = image_embedding.shape[0]
        image_embeddings[idx : idx + batch_size] = image_embedding
        text_embeddings[idx : idx + batch_size] = text_embedding
        idx += batch_size

    if return_numpy:
        return image_embeddings.cpu().numpy(), text_embeddings.cpu().numpy()
    else:
        return image_embeddings, text_embeddings




# torch cosine similarity
def sim_with_embeddings(embeddings, query_embedding):
    return F.cosine_similarity(embeddings, query_embedding, dim=-1)



# unfreeze the layers we want 
def freeze_clip_projection_layers(model,unfreeze_transformer_last=False):
    for param in model.parameters():
        param.requires_grad= False
        
    # unfreeze the models.final same space projection 
    model.visual.proj.requires_grad = True
    model.text_projection.requires_grad= True

    # IF user wants to also train the final transformer's layers 
    # TODO: also test training other layers as well 
    # unfreeze the transformers' final projection and normalization layers 
    # maybe later also the mlp , final ln, token_embedding

    # unfreeze transformer final layer 
    if(unfreeze_transformer_last):
        for param in model.transformer.resblocks[-1].mlp.c_proj.parameters():
            param.requires_grad = True

        for param in model.visual.transformer.resblocks[-1].mlp.c_proj.parameters():
            param.requires_grad = True

        # unfreeze normalization layers of final transformer layers 
        for param in model.visual.transformer.resblocks[-1].ln_2.parameters():
            param.requires_grad = True

        for param in model.transformer.resblocks[-1].ln_2.parameters():
            param.requires_grad = True
        
    return model 

def freeze_last_k_transformer_layers(model,last_k):
    for param in model.parameters():
        param.requires_grad= False
        
    # unfreeze the models.final same space projection 
    model.visual.proj.requires_grad = True
    model.text_projection.requires_grad= True

    for param in model.transformer.resblocks[-last_k:].parameters():
        param.requires_grad = True

    for param in model.visual.transformer.resblocks[-last_k:].parameters():
        param.requires_grad = True

    
        
    return model 





# function to show trainable layers 
def print_only_trainable(model):
    print('='.join([('') for _ in range(55)]))
    print('Trainable layers')
    print('='.join([('') for _ in range(55)]))
    for x,(name,_) in zip(model.parameters(),model.named_parameters()):
        if(x.requires_grad):
            print(name,'--->',x.shape)
    print('='.join([('') for _ in range(55)])+'\n')



# function to pick hyperparams for last transformer layers and last projection layers
# only for last transformer + projection layers  (projection components of last layers)
# (only projection components)
def pick_hyperaparameters(model,unfreeze_transformer_last=False):
    optimizer = optim.Adam([
        {'params': model.visual.proj, 'lr': 1e-5,'weight_decay': 1e-4},
        {'params': model.text_projection, 'lr': 1e-5,'weight_decay': 1e-4},
    ])

    if(unfreeze_transformer_last):
        optimizer.add_param_group({'params': model.transformer.resblocks[-1].mlp.c_proj.parameters(), 'lr': 1e-5,'weight_decay': 1e-4})
        optimizer.add_param_group({'params': model.transformer.resblocks[-1].ln_2.parameters(), 'lr': 1e-5,'weight_decay': 1e-4})
        optimizer.add_param_group({'params': model.visual.transformer.resblocks[-1].mlp.c_proj.parameters(), 'lr': 1e-5,'weight_decay': 1e-4})
        optimizer.add_param_group({'params': model.visual.transformer.resblocks[-1].ln_2.parameters(), 'lr': 1e-5,'weight_decay': 1e-4})

    return optimizer 


# function to pick hyperparams of last_k transformer layers (the whole layers)
# and hyperparams for the whole final projection layers 
# (whole layer)
def pick_hyperaparameters_last_k_transformer_layers(model,last_k):
    optimizer = optim.Adam([
        {'params': model.visual.proj, 'lr': 1e-5,'weight_decay': 1e-4},
        {'params': model.text_projection, 'lr': 1e-5,'weight_decay': 1e-4},
    ])

    # Add parameter groups for the last k layers of the vision transformer
    for layer in model.visual.transformer.resblocks[-last_k:]:
        optimizer.add_param_group({'params': layer.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4})
        

    # Add parameter groups for the last k layers of the text transformer
    for layer in model.transformer.resblocks[-last_k:]:
       optimizer.add_param_group({'params': layer.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4})
        
     
    return optimizer 



#-------------------functions specific for gpu training clip ------------------#
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

        
# functio to convert dataframe back to compatible dataset 
def df_to_dataset(df):
    # Convert the DataFrame to a dictionary
    df_dict = df.to_dict(orient='list')
    
    # Ensure the columns are in the correct format
    dataset = hugda.from_dict(df_dict)
    
    # Cast columns to appropriate types (if necessary)
    if 'image' in dataset.column_names:
        dataset = dataset.cast_column('image', dataset.features['image'])
    if 'text' in dataset.column_names:
        dataset = dataset.cast_column('text', dataset.features['text'])
    if 'label' in dataset.column_names:
        dataset = dataset.cast_column('label', dataset.features['label'])
    
    return dataset



# Function to save a checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)

# Function to load a checkpoint
def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0
    

def print_parameter_status(model):
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

# function to plot the training vs validation loss 
def plot_losses(train_losses, val_losses, path):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Contrastive Loss')
    plt.plot(epochs, val_losses, label='Validation Contrastive Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Constrastive Loss')
    plt.xticks(epochs) 
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


#---------------- Following are the augmentation specific classes ------------#

#img_repf = 5
#original_dataset = '.../train_deepl.csv'
#cap_repf = 5


# class to generate 5 x 5 image-caption pairs per image-caption pair according to paper
class DatasetAugmentGenerator:

    # 1
    def __init__(self,
                 augs,
                 savepath='/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/trainimages',
                 final_path2 = '/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/augmented_train.csv',
                 starting_path='/home/ekotsis/s4a/training_and_evaluation/Datasets/augmented/train_deepl.csv',
                 img_repf=5,
                 cap_repf=5):
        self.data = pd.read_csv(starting_path)
        self.img_repf = img_repf
        self.cap_repf = cap_repf
        # the list of augmentation models to augment captions 
        self.augs = augs
        self.savepath = savepath
        self.final_path2=final_path2

    def image_augment(self,data,target_height,target_width):
        data2 = data.copy()
        # path new to save and save augmented image as original_name + index(i)
        #savepath = '/home/ko_st/Documents/Dsit_Thesis/Sample_Datasets/trainimages'
        if(not(os.path.exists(self.savepath))):
            os.makedirs(self.savepath)

        for i,path in enumerate(data2['image']):
            # transforms for the image 
            transforms = T.Compose([
                #T.RandomVerticalFlip(0.5),
                T.RandomResizedCrop( (224,224), scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5),
                #T.RandomRotation((0,180)),
                #T.RandomPerspective(0.5),
                # maybe center crop 
                #T.CenterCrop((target_height,target_width)), # will probably use 2000 
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                #-----------original--------------#
                #T.ConvertImageDtypes(torch.float32),  # Normalize expects float input 
                #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #---------------------------------#
                T.ToTensor(),  # Convert PIL image to a tensor before further transforms
                T.ConvertImageDtype(torch.float32),  # Optional: ensure the image is in float32 type
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])    

            # open pil image and transform 
            img2 = Image.open(path)
            img2 = transforms(img2) 
            name1 = path.split('/')[-1].split('.')[0] + '-' + str(i) + '.jpg'

            #------- transform tensor back to pil and save image ----#
            # reverse normalization
            img2 = img2 * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)  # Un-do std
            img2 = img2 + torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)  # Un-do mean

            img2 = torch.clamp(img2, 0, 1)  # Ensure values are in range [0, 1]
    
            # Convert back to PIL image
            img2 = T.ToPILImage()(img2)

            img2.save(os.path.join(self.savepath,name1)) 

            print(f"Augment image: {path.split('/')[-1].split('.')[0]} --- New image: {i}")
            # replace new name in column 'image'
            data2.loc[i,'image']= os.path.join(self.savepath,name1)
        
        # return the transformed dataframe 
        return data2



    def caption_augment_helper(self,x):
        return [ ''.join(y.augment(x)) for y in self.augs]
    
    # 2
    def image_replicate_augment(self,new_height=1000,new_width=1000):
        data = self.data[['ID','issue','image','issue2','category','issue','issue_source']]
        data = self.data.sort_values(by=['ID'])
        data.reset_index(drop=True)
        self.data = data

        # image_replicas
        replicated_df = np.tile(self.data[0:].values,(self.img_repf,1))

        new_df = pd.DataFrame(replicated_df,columns=self.data.columns)
        
        new_df_sorted = new_df.sort_values(by='ID')
        new_df_sorted.reset_index(drop=True,inplace=True)
        self.new_df_sorted = new_df_sorted


        # Now we have all 5 copies of each image and we augment them 
        # change the resolution to 224,224 or higher 
        self.augmented_data = self.image_augment(self.new_df_sorted,new_height,new_width)
        
    # 3
    # default augs is the list of the default backtranslating classes 
    def caption_augment(self): 
        # TODO : do we use drop=True or not 
        print("caption augment columns are : ",self.data.columns)
        self.data3 = self.data['issue2'].map(lambda x : self.caption_augment_helper(x))
        self.data3.drop(columns=['Unnamed: 0.1','Unnamed: 0'],errors='ignore')
        self.data3.reset_index(drop=True,inplace=True)
        print('it worked')

    # 4
    def caption_replicate_augment(self):
        replicated_df2 = np.tile(self.augmented_data[0:].values,(self.cap_repf,1))
        new_df2 = pd.DataFrame(replicated_df2,columns=self.new_df_sorted.columns)
        self.new_df_sorted2 = new_df2.sort_values(by=['ID','image']).reset_index(drop=True)
        

    # 5
    # bs is given as a factor of img_repf x cap_repf,default  is 25 (5x5)
    def generate_df(self):
        bs = self.img_repf * self.cap_repf
        factor = self.img_repf
        final_df = self.new_df_sorted2.copy()
        for i in range(len(self.data3)) :
            # is this correct ? maybe bs*i : bs*i + bs
            final_df.loc[bs*i:bs*i+bs-1,'issue2'] = factor*self.data3[i]
        #self.final_path = final_path
        self.final_df = final_df
        self.final_df.drop(columns=['Unnamed: 0','Unnamed: 0.1'],errors='ignore',inplace=True)
        self.final_df.reset_index()

    # 6
    # in the end we need to copy all the images duplicates and rename them so that the 
    # hugging face dataset can be generated correctly 
    # we do this in the directory independently of the dataframe and then fix the dataframe naming 
    def final_fix_naming_duplicate(self):
        for i,filename in enumerate(os.listdir(self.savepath)):
            
            for i in range(0,self.img_repf):  # 1 to 5 for the five duplicates
                new_filename = ''.join(filename.split('.')[:-1]) + f'-{i}' + '.jpg' 
                shutil.copy(os.path.join(self.savepath, filename), os.path.join(self.savepath, new_filename))

            # Delete the original image
            os.remove(os.path.join(self.savepath, filename))

    
    # this function adds the original unaugmented images , mapping them to the augmented captions
    # here we feed the same savepath as the self.save_path to save them all together 
    def map_original_images_to_captions(self):
        data_orig = self.data.copy()
        data_orig = data_orig.sort_values(by=['ID'])
        data_orig.reset_index(drop=True,inplace=True)
    

        # image_replicas
        replicated_df = np.tile(data_orig.values,(self.img_repf,1))

        new_df = pd.DataFrame(replicated_df,columns=data_orig.columns)
        
        new_df_sorted = new_df.sort_values(by='ID')
        new_df_sorted.reset_index(drop=True,inplace=True)
        new_df_mapped = new_df_sorted.copy()
        cnt = 0
        for i in range(len(new_df_sorted)):
                if(cnt==0):
                    new_df_mapped.loc[i,'issue2'] = self.augs[cnt].augment(new_df_sorted.loc[i,'issue2'])
                else :
                    new_df_mapped.loc[i,'issue2'] ='.'.join(self.augs[cnt].augment(new_df_sorted.loc[i,'issue2']))
                if((i+1) % self.img_repf == 0):
                    cnt=0
                else :
                    cnt+=1
                          

        # having mapped the captions to the the same image 
        # we copy each 5-pair-rows/image to the new location and also change the location in the dataframe 'image'
        
        for i in range(len(new_df_mapped)):
        
            name1 = new_df_mapped.loc[i,'image'].split('/')[-1].split('.')[0] + '-' + 'ori' + '-' + str(i) + '.jpg'
            
            shutil.copy(os.path.join(new_df_mapped.loc[i,'image']), os.path.join(self.savepath, name1))
            
            # replace also the naming convention in the dataset 
            new_df_mapped.loc[i,'image']= os.path.join(self.savepath,name1)

        # save the original modified dataset to concat it later to the augmented dataset 
        self.new_df_mapped = new_df_mapped

        #new_df_mapped.to_csv('/home/ko_st/Documents/Dsit_Thesis/Sample_Datasets/original_mappings/orig.csv')
        #self.mapped_data.to_csv('/home/ko_st/Documents/Dsit_Thesis/Sample_Datasets/original_mappings/orig.csv')
        




    # 7 fix the naming scheme in the dataframe as well
    def final_fix_dataframe_naming(self):
        indexer = 0
        for i in range(len(self.final_df)):
                if(i % self.img_repf == 0 and i!=0):
                    indexer = 0
                self.final_df.loc[i,'image'] = ''.join(self.final_df.loc[i,'image'].split('.')[:-1]) +f'-{indexer}' + '.jpg' 
                # not the first index and factor of 5 
                indexer += 1

        # now that we have fixed the naming scheme and duplicated the images 
        # we concatenate the original dataset in order to have the original images as well 
        #TODO :we probably should map all 5 captions to each of the original images 
        # we should concat after we fix the mapping for the original dataset 
        self.new_df_mapped.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True,errors='ignore')
        self.final_df2 = pd.concat([self.final_df,self.new_df_mapped],axis=0)     

        # write final df 
        self.final_df2.to_csv(self.final_path2)
                    


    # call function to implement the whole pipeline
    def __call__(self,augs):
        self.image_replicate_augment()
        print("finished image augmentation")
        self.caption_augment()
        print("finished caption augmentation")
        self.caption_replicate_augment()
        print("finished caption replicating")
        self.generate_df()
        print("finished generating dataaset")
        self.final_fix_naming_duplicate()
        print("finished mapping dataset")
        # mapping the original images to all the augmented captions
        self.map_original_images_to_captions()
        print("finished mapping originals dataset")
        self.final_fix_dataframe_naming()
        print("finished naming original dataset")


#----------------------- Postprocessing the augmented dataset (remove duplicate words, wrong captions)--#
class DuplicateCleaner:
    def __init__(self,data_path):
        data = pd.read_csv(data_path)
        self.data = data.copy()
        self.v2 = []
    

    def countDuplicate(self,s1,i):
        v1 ={}
        def countWords(it1,i):
            if(it1 not in v1):
                v1[it1] = 1 
            else:
                v1[it1] += 1
            if(v1[it1] > 20): 
                self.v2.append(i)

            
        for x in list(word_tokenize(s1)):
            countWords(x,i)
        

    def __call__(self):
        for i in range(len(self.data['issue2'])):
            # add self.v2 in countDuplicate
            self.countDuplicate(self.data.loc[i,'issue2'],i)
        duplicate_index = list(set(self.v2))
        data_clean = self.data.copy()


        temp_images_list =[]
        # remove the images that correspond to those indexes from the trainimages/path
        for x in duplicate_index:
            temp_images_list.append(data_clean.loc[x,'image'])
        # delete the images
        for x in temp_images_list:
            os.remove(x)
        
        # drop the rows from the dataset
        data_clean.drop(duplicate_index,inplace=True)
        data_clean.reset_index(drop=True,inplace=True)
        data_clean.drop(columns=['Unnamed: 0','Unnamed: 0.2','Unnamed: 0.1'],inplace=True,errors='ignore')
        #print(temp_images_list)
        
        return data_clean
    
         


#--------------------------- Following is the pipeline to detect and remove handwritten images------#
# function to detect the handwritten images in the dataset and drop them (of course does not work 100%)

# Function to detect horizontal and vertical lines
def detect_lines(image):

    image = cv2.imread(image)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,gray_thresh = cv2.threshold(gray,160, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_thresh, 150,255 ,apertureSize=3,)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=451, maxLineGap=40)
    
    horizontal_lines = []
    vertical_lines = []
    

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # Horizontal line
                horizontal_lines.append(line[0])
            elif abs(x1 - x2) < 10:  # Vertical line
                vertical_lines.append(line[0])

    if(len(horizontal_lines) > 10) or (len(vertical_lines)>10):
        return True
    
    return False

# function to apply the previous to a whole dataset 
def mark_all_handwritten(path_to_csv):
    handwritten_images = []
    dataset = pd.read_csv(path_to_csv)
    for i in range(len(dataset)):
        image_path = dataset.loc[i,'image']
        if(detect_lines(image_path)):
            handwritten_images.append(image_path)
    return handwritten_images


# function to copy the handwritten images to separate folder 
def separate_handwritten(handwritten_images_list,path_to_copy):
   if(not(os.path.exists(path_to_copy))):
      os.makedirs(path_to_copy)

   for x in handwritten_images_list:
      shutil.copy(x,os.path.join(path_to_copy,x.split('/')[-1]))
   print(' Handwritten images copied')

# function to remove all image-caption pairs existing inside the "separate folder"
# from the csv files so we can generate the dataset without them 
   
def remove_handwritten_from_csv(path_to_csv,path_to_handwritten):
    l1 =[]
    for x in glob.glob(os.path.join(path_to_handwritten, '*.jpg')):
        l1.append(x.split('/')[-1])

    p1 = pd.read_csv(path_to_csv)
    
    # mask to drop the handwritten image caption pairs 
    p2 = p1['image'].map(lambda x : x.split('/')[-1] in (l1) )

    # drop the handwritten from the csv 
    p3 = p1[~p2].reset_index(drop=True)

    # new dataframe without them 
    return p3 
    

