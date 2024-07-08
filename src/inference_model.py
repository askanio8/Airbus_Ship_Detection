import os
from typing import Tuple

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# Function to decode RLE to a mask
def rle_decode(mask_rle: str, shape: Tuple[int, int] = (768, 768)) -> np.ndarray:
    s = list(map(int, mask_rle.split()))
    starts, lengths = [np.asarray(x) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# Function to load and preprocess the image
def load_image(img_path: str, img_size: Tuple[int, int] = (768, 768)) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img


# Paths to the data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'src', 'unet_model.h5')
img_path = os.path.join(base_dir, 'src','4c5bdcd42.jpg')

# Load the model
model = tf.keras.models.load_model(model_path)

# Load and preprocess the image
img = load_image(img_path)

# Perform prediction
predicted_mask = model.predict(img)

# Postprocess the mask for display
predicted_mask = predicted_mask.reshape(768, 768)

# Display the image and predicted mask
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img.reshape(768, 768), cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(predicted_mask, cmap='gray')
ax[1].set_title('Predicted Mask')
ax[1].axis('off')

plt.show()
