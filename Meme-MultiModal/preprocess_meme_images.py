
import argparse
import json
import os
from tqdm import tqdm
import cv2
import pytesseract
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from torch.nn.functional import softmax

from utils.dataset.multi_category_memotion import MultiCategoryeMemotionEvalDataset


def get_text_from_meme(meme_dir_path):
    """
    :param meme_dir_path: path to the meme directory
    :return: a list of dictionaries with the following format
    # meme_text_list = [
    #     {"meme_filename": "image_ (1093).jpg", "text": "OMG?!They told me this was a solo protrait!"},
    #     {"meme_filename": "image_ (109).jpg", "text": "NEXT STOP: KNOWLEDGE"}
    # ]
    """
    image_list = os.listdir(meme_dir_path)
    allowed_format = ["png", "jpg", "jpeg"]
    for image in image_list:
        format = image.split(".")[-1].lower()
        if format not in allowed_format:
            image_list.remove(image)
    temporary_image_folder = "temporary_folder"
    if not os.path.exists(temporary_image_folder):
        os.mkdir(temporary_image_folder)
    meme_text_list = []
    for image in image_list:
        curr_dict = {}
        curr_dict["meme_filename"] = image
        image_name = image.split(".")[0]
        img_path = os.path.join(meme_dir_path, image)
        print("read in", img_path)
        img = cv2.imread(img_path)
        bilateral_blur = cv2.bilateralFilter(img, 5, 55, 60)
        grayscale = cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(grayscale, 240, 255, 1)
        temp_image_path = os.path.join(temporary_image_folder, f"{image_name}.png")
        cv2.imwrite(temp_image_path, im)
        ocr_text = pytesseract.image_to_string(temp_image_path).replace("\n", " ")
        curr_dict["text"] = ocr_text
        meme_text_list.append(curr_dict)
    # shutil.rmtree(temporary_image_folder)
    return meme_text_list


index_2_sentiment = ["happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]

new_index_2_sentiment = ["joy", "anger", "sorrow", "fear", "hate", "surprise"]

activation = {}

@torch.no_grad()
def get_activation(name1, name2):
    def hook(model, input, output):
        activation[name1] = output.last_hidden_state.detach()
        activation[name2] = output.pooler_output.detach()
    return hook

@torch.no_grad()
def get_memes_to_prob_dist_map(args, device, meme_text_list: list):
    """
    Use pretrained multi category meme model to process meme images with their texts
    :param args
    :param meme_text_list: [{"meme_filename": "image_ (0).jpg", "text": "\"That moment after you throw up and your friend asks you \"\"YOU GOOD BRO?\"\" I'M FUCKIN LIT"}]
    :return meme_filename_to_prob_dist: dict (map from meme filename to its probability distribution over 7 sentiment classes)
    """
    print("Start processing meme images")
    # Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    model = torch.load(os.path.join(args.model_dir, args.pretrained_multi_category_meme_model),
                       map_location=torch.device(device))
    bert = torch.load(os.path.join(args.model_dir, args.pretrained_bert), map_location=torch.device(device))
    # print(model)
    feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', args.cnn_model, weights='VGG16_Weights.IMAGENET1K_V1')

    # Init dataset
    dataset = MultiCategoryeMemotionEvalDataset(
        args.meme_image_dir,
        meme_text_list,
        args.max_token_length,
        tokenizer,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, generator=torch.Generator(device=device))
    meme_filename_to_prob_dist = {}
    for data_batch in tqdm(dataset_loader):
        meme_filenames = data_batch["meme_filename"]
        input_ids = data_batch["input_ids"]
        images = data_batch['image']
        attention_mask = data_batch['attention_mask']

        if device == 'cuda':
            with torch.cuda.device(0):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()
                images = images.cuda()

        if images.size()[0] != input_ids.size()[0]:
            continue

        feature_images = feature_extractor.features(images)
        feature_images = feature_extractor.avgpool(feature_images)
        feature_images = torch.flatten(feature_images, 1)
        feature_images = feature_extractor.classifier[0](feature_images)

        bert.bert.register_forward_hook(get_activation('last', 'pool'))
        outputs = bert(input_ids, attention_mask)

        outputs = softmax(model(
            last_hidden=activation['last'],
            pooled_output=activation['pool'],
            feature_images=feature_images
        ), dim=1).tolist()

        # Collect the results into dictionary
        # print(meme_filenames)
        # print(outputs)
        # _, preds = torch.max(outputs, dim=1)
        # for pred in preds:
        #     print(index_2_sentiment[pred])
        for idx, filename in enumerate(meme_filenames):
            prob_dist = outputs[idx]
            # combine happiness and love to one class (joy)
            prob_dist[1] = prob_dist[0] + prob_dist[1]
            meme_filename_to_prob_dist[filename] = prob_dist[1:]

        with open("data/meme_filename_to_prob_dist.json", "w") as outfile:
            json.dump(meme_filename_to_prob_dist, outfile)


def preprocess_meme_images(meme_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme_image_dir", type=str, default=meme_dir,
                        help="where meme images are stored")

    parser.add_argument("--model_dir", type=str, default="pretrained_models",
                        help="where pretrained model is stored")
    parser.add_argument("--pretrained_multi_category_meme_model", type=str, default="multi_category_meme_model.pt",
                        help="filename of pretrained multi category meme model")
    parser.add_argument("--pretrained_bert", type=str, default="pretrained_albert.pt",
                        help="filename of pretrained bert model")

    parser.add_argument("--max_token_length", type=int, default=50,
                        help="max token length of meme text")
    parser.add_argument('--cnn_model', type=str, default="vgg16",
                        help='pretrained CNN to use for image feature extraction')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    meme_text_list = get_text_from_meme(args.meme_image_dir)
    print("meme_text_list: ", meme_text_list)
    # TODO 1. preprocess meme images
    # 2. Preprocess images and get probability distributions
    get_memes_to_prob_dist_map(args, 'cpu', meme_text_list)
    print("finish preprocessing")


if __name__ == "__main__":
    #TODO: change directory if run from script
    preprocess_meme_images("data/memes/")
