import os

import numpy
import torch
import pandas as pd
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-defined hypaparameters
Image.MAX_IMAGE_PIXELS = 1000000000
dataset = 'multi_category_memotion_dataset'

sentiment_2_index = {
    "happiness": 1,
    "love": 2,
    "anger": 3,
    "sorrow": 4,
    "fear": 5,
    "hate": 6,
    "surprise": 7
}
index_2_sentiment = ["", "happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]


class MultiCategoryeMemotionDataset(Dataset):

    def __init__(self, root_dir, dataframe, max_len, tokenizer, transform=None):
        # dataframe = {"filepath": [], "text": [], "label": []}
        self.data_dict = dataframe
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        # BERT tokenizer
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + 'images/' + self.data_dict.loc[idx, "filepath"]
        image = Image.open(img_name).convert('RGB')
        label = self.data_dict.loc[idx, "label"]

        text = str(self.data_dict.loc[idx, "text"])
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text_encoded['input_ids'].flatten(),
                  'attention_mask': text_encoded['attention_mask'].flatten(),
                  "label": torch.tensor(label, dtype=torch.long)}
        return sample


class MultiCategoryeMemotionEvalDataset(Dataset):

    def __init__(self, root_dir, dataframe, max_len, tokenizer, transform=None):
        # dataframe = {"meme_filename": [], "text": []}
        self.data_list = dataframe
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        # BERT tokenizer
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + self.data_list[idx]["meme_filename"]
        image = Image.open(img_name).convert('RGB')

        text = str(self.data_list[idx]["text"])
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        if self.transform:
            image = self.transform(image)

        sample = {'meme_filename': self.data_list[idx]["meme_filename"],
                  'image': image,
                  'input_ids': text_encoded['input_ids'].flatten(),
                  'attention_mask': text_encoded['attention_mask'].flatten()}
        return sample