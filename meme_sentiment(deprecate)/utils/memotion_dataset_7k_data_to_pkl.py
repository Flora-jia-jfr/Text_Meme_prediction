"""
Parse 3 class meme data to pickle file
"""
import os
from preprocessing import preprocess_txt
import pickle
import pandas as pd
import numpy as np

data = {"filepath": [], "text": [], "label": []}
dirname = "data/memotion_dataset_7k/"
image_dirname = "images/"
output_pkl_filename = "memotion_dataset_7k.pkl"

sentiment_2_index = {
    "very_negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very_positive": 4
}

n_sentiment = 5


def read_data():
    df = pd.read_csv(os.path.join(dirname, "labels.csv"), encoding='latin1')
    n_data = len(df.index)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"INFO: read in data {index}/{n_data}")
        filepath = os.path.join(dirname, image_dirname, row["image_name"])
        data['filepath'].append(filepath)
        data['text'].append(preprocess_txt(str(row['text_corrected'])))
        data['label'].append(np.array([sentiment_2_index[row['overall_sentiment']]]))


    # Store to local pickle file
    with open(os.path.join(dirname, output_pkl_filename), "wb") as pickle_out:
        pickle.dump(data, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    read_data()