import os

from PIL import Image
import json
import numpy as np
from torch.nn.functional import kl_div
import torch
from transformers import pipeline

ekman_map = {
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'nervousness': 'fear',
    'joy': 'joy',
    'amusement': 'joy',
    'approval': 'joy',
    'excitement': 'joy',
    'gratitude': 'joy',
    'love': 'joy',
    'optimism': 'joy',
    'relief': 'joy',
    'pride': 'joy',
    'admiration': 'joy',
    'desire': 'joy',
    'caring': 'joy',
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',
    'surprise': 'surprise',
    'realization': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    'neutral': 'neutral'
}


def get_emotion_distribution_from_context(context):
    # tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    # model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa', return_all_scores=True)
    # print("type of context: ", type(context))
    result_dict = emotion(context)[0]
    # result_dict.sort(key=lambda item: item['score'], reverse=True)
    ekman_result = {
    "joy": 0,
    "anger": 0,
    "sadness": 0,
    "fear": 0,
    "disgust": 0,
    "surprise": 0,
    "neutral": 0
    }
    for emotion in result_dict:
        detailed_label = emotion['label']
        ekman_label = ekman_map[detailed_label]
        ekman_result[ekman_label] += emotion['score']
    # print(ekman_result)
    return list(ekman_result.values())


def get_top_k_images(text_prob_distribution: list, top_k_num: int, meme_filename_to_prob_dist: dict,
                     confidence_threshold=0.8):
    """
    Get top k images based on input text and meme prob distributions
    :param text_prob_distribution: dialogue text distribution over 7 sentiment classes (TODO after softmax?)
    :param top_k_num: upper bound of k
    :param meme_filename_to_prob_dist: map from meme filename to its probability distribution over 7 sentiment classes
    :param confidence_threshold will take images of only the argmax(prob) emotion class if the probability exceeds the threshold value

    :return image_filename_list_result list containing filename of top k images sorted in descending order
    """

    assert len(text_prob_distribution) == 7

    max_text_emotion_prob = max(text_prob_distribution)
    max_text_emotion_index = np.argmax(text_prob_distribution)

    if max_text_emotion_index == 6:  # highest is neutral
        return None
    else:
        neutral_score = text_prob_distribution[-1]
        print("neutral_score", neutral_score)
        print("original: ", text_prob_distribution)
        text_prob_distribution = [score/(1-neutral_score) for score in text_prob_distribution[:6]]
        print("modified: ", text_prob_distribution)

    # Filter by emotion class index and sort by probability of that class
    images_of_max_emotion_class = dict(sorted(
        {filename: prob_dist for filename, prob_dist in meme_filename_to_prob_dist.items() if
         np.argmax(prob_dist) == max_text_emotion_index}.items(),
        key=lambda item: item[1][max_text_emotion_index],
        reverse=True
    ))
    print(images_of_max_emotion_class)
    if max_text_emotion_prob >= confidence_threshold:
        image_filename_list_result = list(images_of_max_emotion_class.keys())
    else:
        image_filename_list_result = sorted(images_of_max_emotion_class.items(),
                                            key=lambda item: kl_div(torch.tensor(item[1][max_text_emotion_index]),
                                                                    torch.tensor(text_prob_distribution)), reverse=True)

    return image_filename_list_result[:top_k_num]

def get_image_list(context, k):
    """
    get the list of images that match the context best
    :return:
    """
    with open("data/meme_filename_to_prob_dist.json", "r") as f:
        meme_filename_to_prob_dist = json.load(f)
    # print(meme_filename_to_prob_dist)
    # TODO 3. Run terminal command ui
    # happiness", "love", "anger", "sorrow", "fear", "hate", "surprise
    # context = "I feel great. Are you still sad?"
    text_prob = get_emotion_distribution_from_context(context)
    print("text_prob: ", text_prob)

    # TODO: need to let user input
    im_root_path = "data/memes"

    # text_prob = [0.6, 0.3, 0, 0, 0, 0, 0.1]
    images = get_top_k_images(text_prob, k, meme_filename_to_prob_dist)
    # print("images", images)
    # for im in images:
    #     im_path = os.path.join(im_root_path, im)
    #     im = Image.open(im_path)
    #     im.show()
    return images


if __name__ == "__main__":
    context = "I feel great. Are you still sad?"
    get_image_list(context, 10)