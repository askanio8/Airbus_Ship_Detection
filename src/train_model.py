import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, Concatenate, UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple


# Define the Dice Score metric
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), dtype=tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)


# Paths to the data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_folder = os.path.join(base_dir, 'airbus-ship-detection-subset', 'train_v2_subset')
csv_path = os.path.join(base_dir, 'airbus-ship-detection-subset', 'train_ship_segmentations_v2_subset.csv')


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


# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_folder: str, csv_path: str, batch_size: int = 10, img_size: Tuple[int, int] = (768, 768)):
        self.img_folder = img_folder
        self.data = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_files = os.listdir(img_folder)
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.img_files) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_files = self.img_files[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__data_generation(batch_files)
        return np.array(images), np.array(masks)

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.img_files)

    def __data_generation(self, batch_files: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        images = []
        masks = []
        for img_file in batch_files:
            img_path = os.path.join(self.img_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0

            mask = np.zeros(self.img_size, dtype=np.uint8)
            mask_rles = self.data[self.data['ImageId'] == img_file]['EncodedPixels'].values
            for mask_rle in mask_rles:
                if isinstance(mask_rle, str):
                    mask += rle_decode(mask_rle, self.img_size)

            mask = np.clip(mask, 0, 1)

            images.append(img.reshape(self.img_size[0], self.img_size[1], 1))
            masks.append(mask.reshape(self.img_size[0], self.img_size[1], 1))

        return images, masks


# Define a lightweight U-Net architecture
def unet(input_size: Tuple[int, int, int] = (768, 768, 1)) -> Model:
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Create and compile the model
model = unet()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_coefficient])

# Data generator
batch_size = 10
train_generator = DataGenerator(img_folder, csv_path, batch_size=batch_size)

# Train the model
model.fit(train_generator, epochs=1, steps_per_epoch=len(train_generator), validation_data=train_generator,
          validation_steps=len(train_generator) // 10)

# Save the model
model.save('unet_model.h5')


