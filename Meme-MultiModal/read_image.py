import os
import cv2
import pytesseract
import shutil


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
    temporary_image_folder = "temporary_folder"
    if not os.path.exists(temporary_image_folder):
        os.mkdir(temporary_image_folder)
    meme_text_list = []
    for image in image_list:
        curr_dict = {}
        curr_dict["meme_filename"] = image
        image_name = image.split(".")[0]
        img_path = os.path.join(meme_dir_path, image)
        img = cv2.imread(img_path)
        bilateral_blur = cv2.bilateralFilter(img, 5, 55, 60)
        grayscale = cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(grayscale, 240, 255, 1)
        temp_image_path = os.path.join(temporary_image_folder, f"{image_name}.png")
        cv2.imwrite(temp_image_path, im)
        ocr_text = pytesseract.image_to_string(temp_image_path).replace("\n", " ")
        curr_dict["text"] = ocr_text
        meme_text_list.append(curr_dict)
    shutil.rmtree(temporary_image_folder)
    return meme_text_list


meme_text_list = get_text_from_meme("memes")
print(meme_text_list)
