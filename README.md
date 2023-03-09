# EMeme
CSCI499 Final Project: EMeme

> Author: Furong(Flora) Jia & Tianyi(Lorena) Yan

## Clone repo & Install requirements:
```
git clone git@github.com:Lorenayannnnn/CSCI-499-Text-Meme-Prediction.git
pip install -r requirements.txt
```


## Info
This repo contains 2 parts: 
1. Training models for text-meme prediction
2. Running GUI that allows users to select local directory that contains their own memes, which will be processed by our pretrained models. Users will be prompted to input sentences, and our model will give meme recommendations from their collection.

## Text-meme Model Training
If you want to re-train models, please run:
```
cd Meme-MultiModal
```
and see [README](Meme-MultiModal/README.md) for more details.

## Running GUI
The pretrained models have been stored inside the [pretrained_models](Meme-MultiModal/pretrained_models) directory, so that the user can directly use our program. Please follow the instructions below to run the program:
1. Create a directory and store your meme images under the [data](data) directory.
2. TODO: run `gui.py` to start the GUI
3. Click the **_Select Directory_** button and choose the local directory that stores your memes
4. Click the **_Meme Preprocess_** button. All your meme images will be preprocessed by our model:
   1. pytesseract is used to identify texts inside the meme images
   2. Images and texts are then encoded and passed into our pretrained models to get probability distribution of each meme image over 6 sentiment classes 
   3. Probability distributions will be stored in **_meme_filename_to_prob_dist.json_** file under [data](Meme-MultiModal/data) directory
5. Enter your sentence inside the dialogue box and click the button to get meme recommendations!
