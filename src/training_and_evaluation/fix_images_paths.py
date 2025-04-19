import pandas as pd
import config_files
import os
import json
from datasets import load_dataset
from torch.utils.data import Dataset
import shutil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np


# SOS
# set up stopwords as the set of the stopwords (WE CAN'T ITERATE IF IT IS NOT A SET)
stopwords = set(stopwords.words(config_files.stopwords_set))


# contains utilities and creates the datasets
def json_metadata(path=config_files.path_to_dataset, name="metadata.jsonl", pairs=[]):
    with open(path + name, "w") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")


# class to fix image paths and prepare datasets
class DatasetsPreprocessor:
    def __init__(self, dataset="train_deepl.csv"):
        self._dataset = dataset

    # correct paths in 'image' comumn
    # here we should fix the datasets
    def fix_image_paths(self, path=config_files.path_to_dataset):
        self._change_count = 0

        self._data = pd.read_csv(path + self._dataset)

        # maybe use a copy of the dataframe

        # drop nan values (we remove the rows-samples with empty translation )
        print(
            "\n Missing values in issue column : "
            + str(self._data["issue"].isnull().sum())
        )

        # preprocessing translated captions -- clean nal, fix some translations
        # fix captions
        # drop null issues rows
        self._data.dropna(subset=["issue"], inplace=True)
        print(self._data["issue"])
        # add here the transformations of each caption
        # also drop row 1006

        # ----------------------------------------------#
        # stopwords and pattern substitution
        # CAPTION PROCESSOR APPLIES STOPWORDS AND REGEX IN 'issue2' column
        # so we can use 'issue' to keep the original captions (in metadata.jsonl)
        # or 'issue2' to create the shortened captions (after stopwords/regex)
        # this is done in self._image_caption_pairs, where we pick our column pairs
        # i will give the shortened captions (after regex/stopwords for now)
        # and we need to test the model with different caption options (regex/stopwords/original)
        # IF YOU WANT THE Original captions
        # 1) remove the 2 lines below calling the caption preprocessor
        # 2) replace "issue2" with "issue" in , self._image_caption_pairs=... , in copy_images method below
        caption_preprocessor = CaptionPreprocessor(data=self._data)
        self._data = caption_preprocessor()

        # --------- Post processing ---------#
        # let's also take care of deleting all lines with NaN after replacements
        # finally remove captions with only : ".","","not"
        self._data["issue"].replace("", np.nan, inplace=True)
        self._data["issue2"].replace("", np.nan, inplace=True)
        self._data["issue2"].replace(".", np.nan, inplace=True)
        self._data["issue"].replace(".", np.nan, inplace=True)
        self._data["issue"].replace("not", np.nan, inplace=True)
        self._data["issue2"].replace("not", np.nan, inplace=True)
        self._data.dropna(subset=["issue", "issue2"], inplace=True)
        # ----------------------------------------------#

        # fix image paths
        # self._data["image"].replace(**config_files.path_mapping, inplace=True)

        self._data.drop(columns=["Unnamed: 0"], inplace=True)

        # save again fixed datasets
        self._data.to_csv(path + self._dataset, chunksize=100)

    # copy images into the 'images' directory and create .jsonl files
    def copy_images(
        self, path=config_files.path_to_dataset, add_category=False, images_exist=False
    ):
        # read again the dataset (the following line should be removed if everything works correctly in
        # fix_image_paths)

        self._data = pd.read_csv(path + self._dataset)

        # do the caption preprocessing here (stopwords/regex)
        # self.cap1 = CaptionPreprocessor(self._data)
        # self._data = self.cap1()
        # self._data.to_csv(config_files.path_to_dataset + self._dataset, chunksize=100)

        # select "issue" to kep original captions
        # or "issue2" to use the regex/stopwords preprocessed captions as final captions
        # here i used "issue2"
        # just pick "issue" here if we want the original uncut captions
        # to create the caption-image pairs with the original uncut captions again
        self.add_category = add_category
        if self.add_category:
            self._image_caption_pairs = [
                {
                    "file_name": self._data.loc[i, "image"].split("/")[-1],
                    "text": self._data.loc[i, "issue2"],
                    "label": self._data.loc[i, "category"],
                }
                for i in range(len(self._data))
            ]
        else:
            self._image_caption_pairs = [
                {
                    "file_name": self._data.loc[i, "image"].split("/")[-1],
                    "text": self._data.loc[i, "issue2"],
                }
                for i in range(len(self._data))
            ]

        # create the folder where we will copy the images

        if not os.path.exists(
            os.path.join(path, self._dataset.split("_")[0] + "images")
        ):
            os.makedirs(os.path.join(path, self._dataset.split("_")[0] + "images"))

        # copy image in the directory
        if images_exist == False:
            self._data["image"].map(
                lambda x: shutil.copy(
                    x,
                    os.path.join(path, self._dataset.split("_")[0] + "images"),
                )
            )

        # create jsonl file
        json_metadata(
            path=os.path.join(path, self._dataset.split("_")[0] + "images/"),
            name="metadata.jsonl",
            pairs=self._image_caption_pairs,
        )

    # create dataset for hugging face api
    def create_Dataset(self, path=config_files.path_to_dataset):
        self._dataset = load_dataset(
            "imagefolder",
            data_dir=os.path.join(path, self._dataset.split("_")[0] + "images"),
            split="train",
        )
        return self._dataset


class CaptionPreprocessor:

    def __init__(self, data):
        self.data2 = data.copy()
        # self.data2.drop(columns=["Unnamed: 0"], inplace=True)

    # function to remove whole rows if the caption has a certain pattern (drop rows on string pattern)
    def dropRows(self):
        print(f"Total Lines : {len(self.data2)}\n")
        self.data2 = self.data2[~self.data2["issue"].isin(config_files.drop_patterns)]
        print(f"Total Lines after removal : {len(self.data2)}\n")

    @staticmethod
    def removeStopWords(caption, stopwords=stopwords):
        words = word_tokenize(caption)
        caption_filtered = " ".join([w for w in words if w.lower() not in stopwords])
        return caption_filtered

    # remove stop words and regex
    def applyremoveStopwordsRegex(self):
        self.data2["issue"].replace(
            to_replace=config_files.caption_mapping, inplace=True, regex=True
        )
        self.data2["issue2"] = self.data2["issue"].apply(self.removeStopWords)
        self.data2["issue2"].replace(
            to_replace=config_files.caption_mapping, inplace=True, regex=True
        )
        # self.data2['issue2'].replace(to_replace = {' *. *. *' : ''},regex = True)
        # print(self.data2[:5]["issue2"])
        return self.data2

    # the whole pipeline
    # clean the dataset of rows with non descriptive patterns
    # and apply the remove regex/stopwords
    def __call__(self):
        self.dropRows()
        return self.applyremoveStopwordsRegex()

    # augment the dataset with synonyms and back translation


class UnprocessedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # for x in self.dataset:
        #    inputs = processor(text=["storage room",caption], images=im, return_tensors="pt", padding=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = item["image"]
        text = item["text"]
        return images, text


# dataset class for CLIP model
# for CLIP preprocessing
class ClipDataset(Dataset):
    def __init__(self, dataset, processor, tokenizer):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        text_encoding = self.tokenizer(
            text=[item["text"]], padding="max_length", return_tensors="pt"
        )
        image_encoding = self.processor(images=item["image"], return_tensors="pt")

        """ add the next 2 lines if you want to write the caption tokens in a txt file to further study them"""
        # with open(config_files.path_to_dataset+ "/tokens1.txt", "a") as file1:
        #    file1.write(str(self.processor.tokenizer(text).tokens()) + '\n')

        # removing batch dimension
        encoding = {**text_encoding, **image_encoding}
        return encoding


# class to set up preprocessed dataset for training
# this is the class for GIT preprocessing
class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"],
            text=item["text"],
            padding="max_length",
            return_tensors="pt",
        )

        # removing batch dimension
        encoding = {k: v for k, v in encoding.items()}
        # encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


if __name__ == "__main__":
    # this class fixes the image paths inside the data columns
    # and copies the correct images from the datasets into their corresponding directories

    dp = DatasetsPreprocessor("train_deepl.csv")
    dp.fix_image_paths()
    dp.copy_images()

    dp2 = DatasetsPreprocessor("val_deepl.csv")
    dp2.fix_image_paths()
    dp2.copy_images()

    dp3 = DatasetsPreprocessor("test_deepl.csv")
    dp3.fix_image_paths()
    dp3.copy_images()
