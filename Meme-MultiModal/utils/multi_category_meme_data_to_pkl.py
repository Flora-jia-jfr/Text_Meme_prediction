"""
Parse data to pickle file:
data = {"image": [], "filepath": [], "text": [], "label": []}
"""

import os
import nltk
from preprocessing import preprocess_txt
import pickle
import pandas as pd
import numpy as np

data = {"filepath": [], "text": [], "label": []}
dirname = "../data/Multi_category_Meme"
image_dirname = "images/"
data_filename = "E_text.csv"
label_filename = "label_E.csv"
output_pkl_filename = "multi_category_meme_data.pkl"

sentiment_2_index = {
    "happiness": 1,
    "love": 2,
    "anger": 3,
    "sorrow": 4,
    "fear": 5,
    "hate": 6,
    "surprise": 7
}
# index_2_sentiment = ["", "happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]

n_sentiment = 7


def read_data(image_to_text_filename: str, image_to_label_filename: str):
    image_filename_2_text = {}
    image_filename_2_sentiment = {}

    # Read in text inside each meme image
    image_to_text_df = pd.read_csv(os.path.join(dirname, image_to_text_filename), encoding='latin1')
    n_data = len(image_to_text_df)
    for index, row in image_to_text_df.iterrows():
        image_filename_2_text[row['file_name']] = str(row['text']).strip()

    # Read in sentiment category of each meme image
    image_to_label_df = pd.read_csv(os.path.join(dirname, image_to_label_filename), encoding='latin1')
    for index, row in image_to_label_df.iterrows():
        image_filename_2_sentiment[row['file_name']] = int(row['sentiment category'][0]) - 1

    # Create dataframe and store data into pickle file
    for index, image_filename in enumerate(image_filename_2_text):
        if index % 100 == 0:
            print(f"INFO: read in data {index}/{n_data}")
        # filepath = os.path.join(dirname, image_dirname, image_filename)
        data['filepath'].append(image_filename)
        data['text'].append(preprocess_txt(image_filename_2_text[image_filename]))
        data['label'].append(np.array([image_filename_2_sentiment[image_filename]]))

    # Store to local pickle file
    with open(os.path.join(dirname, output_pkl_filename), "wb") as pickle_out:
        pickle.dump(data, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    read_data(image_to_text_filename=data_filename, image_to_label_filename=label_filename)