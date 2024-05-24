

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import urllib.request
import cv2
from tqdm import tqdm
import pickle
import ssl
import pandas as pd

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a new model with ResNet50 base and GlobalMaxPooling2D layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()

# Function to load and preprocess image from URL
def load_and_preprocess_image_from_url(url):
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            img = np.asarray(bytearray(response.read()), dtype="uint8")
    except Exception as e:
        print(f"Error accessing {url}: {e}")
        return None
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

# Function to extract features from image
def extract_features_from_url(url, model):
    preprocessed_img = load_and_preprocess_image_from_url(url)
    if preprocessed_img is None:
        return None
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Read image URLs from CSV file
df = pd.read_csv('data.csv', header=None)  # Read CSV without headers
image_urls = df[0].tolist()  # Use only the first column

filenames = []
feature_list = []

for url in tqdm(image_urls):
    try:
        features = extract_features_from_url(url, model)
        if features is not None:
            feature_list.append(features)
            filenames.append(url)
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Save features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
