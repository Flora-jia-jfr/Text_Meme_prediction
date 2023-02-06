"""
Class for creating PyTorch MeMotion 3_class_memes
"""

from torch.utils import data
from transformers import ViltProcessor
import pandas as pd
import torch
import numpy as np
from PIL import Image


# Reference: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb#scrollTo=Dl2UsPrTHbtu
class MemotionDataset(data.Dataset):
    """
    MemotionDataset
    """
    # data = {"image": [], "filepath": [], "text": [], "label": []}

    def __init__(self, image_filepaths, texts, labels, processor):
        self.image_filepaths = image_filepaths
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        # get image + text
        image_filepath = self.image_filepaths[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_filepath).convert("RGB")
            encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
            # remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()
            # add labels
            encoding['labels'] = torch.tensor(np.array(label))

            return encoding
        except Exception:
            print(f"Error when reading {image_filepath}")


def load_dataset(pickle_filename: str, prop_train=0.8):
    # data = {"image": [], "filepath": [], "text": [], "label": []}
    df = pd.DataFrame(pd.read_pickle(pickle_filename))
    df_train = df.sample(frac=prop_train)
    df_val = df.drop(df_train.index)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    train_dataset = MemotionDataset(image_filepaths=df_train["filepath"],
                              texts=df_train["text"],
                              labels=df_train["label"],
                              processor=processor)

    val_dataset = MemotionDataset(image_filepaths=df_val["filepath"],
                                    texts=df_val["text"],
                                    labels=df_val["label"],
                                    processor=processor)

    return train_dataset, val_dataset
