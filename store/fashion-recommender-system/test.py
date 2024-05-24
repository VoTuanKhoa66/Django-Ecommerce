import pickle
import numpy as np
from numpy.linalg import norm
import urllib.request
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import ssl

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))



# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a new model with ResNet50 base and GlobalMaxPooling2D layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to load and preprocess image from URL
def load_and_preprocess_image(url):
    # Download image from URL
    with urllib.request.urlopen(url) as response:
        img = np.asarray(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Convert bytes data to image array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    preprocessed_img = preprocess_input(img_array)  # Preprocess input for ResNet50
    return preprocessed_img
    

# Load and preprocess image from URL
url = 'https://sneakerholicvietnam.vn/wp-content/uploads/2021/07/air-jordan-1-mid-light-smoke-grey-554725-092-1.jpg'
preprocessed_img = load_and_preprocess_image(url)

# Extract features from image
result = model.predict(preprocessed_img).flatten()
print(result)
normalized_result = result / norm(result)
print(normalized_result)

# Function to compute Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Implementing KNN from scratch
def knn(feature_list, query_feature, k=5):
    distances = []
    skipped_one = False  # Flag to keep track if one similar URL has been skipped
    for idx, feature in enumerate(feature_list):
        # Check if the filename matches the URL and if we haven't skipped one yet
        if filenames[idx] == url and not skipped_one:
            skipped_one = True
            continue  # Skip this one and move to the next iteration
        dist = euclidean_distance(feature, query_feature)
        distances.append((dist, idx))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get the indices of the k nearest neighbors
    neighbors_indices = [idx for dist, idx in distances[:k]]
    
    return neighbors_indices

# Find nearest neighbors
nearest_neighbors = knn(feature_list, normalized_result, k=5)
nearest_neighbors_urls = [filenames[idx] for idx in nearest_neighbors]
print(nearest_neighbors_urls)

# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()

# Function to display images from URLs
def display_images_from_urls(urls):
    for url in urls:
        try:
            with urllib.request.urlopen(url,context=ssl_context) as response:
                img_data = np.asarray(bytearray(response.read()), dtype="uint8")
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (512, 512))
                cv2.imshow('output', img)
                cv2.waitKey(0)
            else:
                print(f"Cannot read image from URL {url}")
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

# Display nearest neighbor images from URLs
display_images_from_urls([filenames[idx] for idx in nearest_neighbors])
