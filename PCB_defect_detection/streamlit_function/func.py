import streamlit as st
from PIL import Image
from pathlib import Path

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def set_title(file):
    file_name = Path(file).name[3:-3].replace("_", " ")

    st.set_page_config(page_title=file_name, layout="wide")
    st.title(file_name)

def image_resize(img_name, file, hsize):
    path = './streamlit_images/' + Path(file).name[3:-3].lower() + "/" + img_name
    image = Image.open(path)
    w, h = image.size
    w *= hsize / h
    
    st.image(path, width=int(w))

def pcb_yolo():
    original_folder = './rotation_test_resized'
    result_folder = './yolov5/runs/detect/exp'

    original_files = [f for f in os.listdir(original_folder) if f.endswith('.jpg')]

    selected_file = random.choice(original_files)
    selected_original_filepath = os.path.join(original_folder, selected_file)

    original_img = Image.open(selected_original_filepath)
    result_file = os.path.join(result_folder, selected_file)
    result_img = Image.open(result_file)

    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(result_img)
    axes[1].set_title('Result Image')
    axes[1].axis('off')
    st.pyplot(plt)