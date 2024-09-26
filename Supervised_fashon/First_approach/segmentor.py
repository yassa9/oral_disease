import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import os
import random
import time
import pickle

generator = pipeline(
    "mask-generation",
    model="jadechoghari/robustsam-vit-huge",
    device=device,
    points_per_batch=32
)

def derive_mask_path(image_path):
    # Replace 'splitted' with 'masks'
    mask_path = image_path.replace('/splitted/', '/masks/')
    
    # Change file extension to '.npy'
    mask_path = os.path.splitext(mask_path)[0] + '.pkl'
    return mask_path


def process_image(image_path, show_time=True):
    start_time = time.time()
    mask_path = derive_mask_path(image_path)
    if os.path.exists(mask_path):
        print(f"Mask already exists for {image_path}, skipping.")
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f)
        return masks
    else:
        print(f"Generating mask for {image_path}")
        image = Image.open(image_path).convert('RGB')
        masks = generator(image)
        with open(mask_path, 'wb') as f:
            pickle.dump(masks, f)
        print(f"Mask saved at {mask_path}")
        end_time = time.time()
        if show_time:
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = total_time % 60
            print(f"Time taken: {minutes} minutes and {seconds:.2f} seconds")
        return masks
                

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def preview_masked_image(image_path):
    masks = process_image(image_path)
    
    image = Image.open(image_path).convert("RGB")
    
    # Create a plot to show original and masked image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image on the left
    axs[0].imshow(np.array(image))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Display the mask output on the right
    axs[1].imshow(np.array(image))
    for mask in masks["masks"]:
        show_mask(mask, ax=axs[1], random_color=True)
    axs[1].set_title("Mask Output")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
