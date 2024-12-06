from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Define paths
model_path = "/Users/asya/Downloads/CocktailScanner/cocktail_model.keras"
label_map_path = "/Users/asya/Downloads/CocktailScanner/label_map.txt"
test_image_path = "/Users/asya/Downloads/CocktailScanner/Cocktails/Cocktails/Mojito/Mojito1.jpg"

# Debugging: Print paths
print(f"Model Path: {model_path}")
print(f"Label Map Path: {label_map_path}")
print(f"Test Image Path: {test_image_path}")

# Verify the files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"Label map file not found: {label_map_path}")

# Load the model
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load the class labels
try:
    labels = {}
    with open(label_map_path, "r") as f:
        for line in f:
            key, value = line.strip().split(":")
            labels[int(key)] = value
except Exception as e:
    raise RuntimeError(f"Failed to load label map: {e}")

# Function to test an image
def test_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  # Normalize and expand dimensions

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions)  # Confidence of top prediction
    predicted_label = labels[np.argmax(predictions)]  # Get the label

    return predicted_label, confidence

# Test the model with an example image
try:
    label, confidence = test_image(test_image_path)
    print(f"Predicted Label: {label}, Confidence: {confidence:.2f}")
except Exception as e:
    print(f"Error: {e}")
