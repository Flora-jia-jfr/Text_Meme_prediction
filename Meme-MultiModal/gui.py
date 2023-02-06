import datetime
import tkinter as tk
from tkinter import filedialog
import os
import subprocess

import cv2
from PIL import Image, ImageTk

from preprocess_meme_images import preprocess_meme_images
from run_terminal_ui import get_image_list

# dir_name = "/Users/florajia/Desktop/memes/"
# image_list = ['6iocfb.jpg', '7k57cf.png']
dir_name = ""
image_list = []
curr_id = 0

root = tk.Tk()
root.geometry('1000x1000+1000+1000')
root.title("Text-Meme-Prediction")

sample_text = tk.Label(root)
sample_text.pack()

selected_dir_label = tk.Label(root, text="")

def UploadAction(event=None):
    global dir_name
    dir_name = filedialog.askdirectory()+"/"
    print('Selected:', dir_name)
    selected_dir_label.config(text=f"Selected: {dir_name}")
    return dir_name

tk.Button(root, text='Select Meme folder', command=UploadAction).place(x=40, y=50)

hint = tk.Label(root, text="")


def preprocess_meme():
    print('dir_name:', dir_name)
    hint.config(text="Preprocessing memes ...")
    if dir_name != "" and dir_name != None:
        print("Preprocessing memes ...")
        try:
            preprocess_meme_images(dir_name)
            hint.config(text="Finish preprocessing memes")
        except cv2.error:
            print("Error happened during preprocessing")
            hint.config(text="Error happened during preprocessing")
    else:
        hint.config(text="Please select a directory for meme first")
    print("Finish preprocessing memes")


tk.Button(root, text="Preprocess meme images", command=preprocess_meme).place(x=40, y=110)

selected_dir_label.place(x=40, y=80)
hint.place(x=40, y=140)

tk.Label(root, text="Enter a sentence here to find its matching memes (Eg: Nice day)").place(x=400, y=50)
input_context = tk.Text(root, height=0.6, width=40, font=("Helvetica", 15))
input_context.place(x=400, y=80)

tk.Label(root, text="Enter the number of memes that your want (Eg: 3)").place(x=400, y=110)
input_k = tk.Text(root, height=0.6, width=40, font=("Helvetica", 15))
input_k.place(x=400, y=140)



def find_top_k_match():
    global image_list, curr_id
    context = input_context.get(1.0, "end-1c")
    k = int(input_k.get(1.0, "end-1c"))
    print("input_context:", context)
    print("k: ", k)
    image_list = get_image_list(context, k)
    print("image_list: ", image_list)
    meme_hint.config(text=f"Meme_list: \n {image_list}", wraplengt=340, anchor=tk.E)
    curr_id = 0
    file_path = dir_name + image_list[curr_id]
    show_memes(file_path)

meme_hint = tk.Label(root, text="")
meme_hint.place(x=40, y=180)

tk.Button(root, text="Matching memes", command=find_top_k_match).place(x=400, y=170)

show_meme_warning = tk.Label(root, text="")
show_meme_warning.place(x=500, y=200)

def show_memes(file_path):
    image = Image.open(file_path)
    width, height = image.size
    ratio = 360/float(width)
    image = image.resize((360, int(height*ratio)))
    width = 360
    height *= ratio
    if height > 580:
        ratio = 580 / float(height)
        image = image.resize((int(width * ratio), 580))
    image = ImageTk.PhotoImage(image)
    imageLabel.configure(image=image)
    imageLabel.image = image
    show_meme_warning.config(text="")

def prev():
    global curr_id
    if curr_id == 0:
        show_meme_warning.config(text="Already the first meme. No previous meme!")
    else:
        show_meme_warning.config(text="")
        curr_id -= 1
        file_path = dir_name + image_list[curr_id]
        show_memes(file_path)

def next():
    global curr_id
    if curr_id == len(image_list)-1:
        show_meme_warning.config(text="Already the last meme. No next meme!")
    else:
        show_meme_warning.config(text="")
        curr_id += 1
        file_path = dir_name + image_list[curr_id]
        show_memes(file_path)


tk.Button(root, text="Previous", command=prev).place(x=400, y=200)
tk.Button(root, text="Next", command=next).place(x=780, y=200)

imageLabel = tk.Label(root)
imageLabel.place(x=400, y=230)




# load_vid_btn.pack()
#
#
# cur_annotation = tk.Label(root, text="")
# cur_annotation.pack()
#
# inputtxt = tk.Text(root, height=2, width=30, borderwidth=3, relief="groove")
# inputtxt.pack()



root.mainloop()
