#This is the main code, which uses implemnted machine learning, created in the file train_model.py
#User needs to install tensorflow


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import requests
import os

# Function to fetch cocktail recipe
def get_cocktail_recipe(cocktail_name):
    API_KEY = '9973533'  # API Key Pro Version
    BASE_URL = f"https://www.thecocktaildb.com/api/json/v2/{API_KEY}/search.php?s={cocktail_name}"
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        if data['drinks']:
            drink = data['drinks'][0]
            instructions = drink['strInstructions']
            drink_thumb = drink['strDrinkThumb']  # Image URL of the cocktail
            return instructions, drink_thumb
    return None, None  # Return None if no recipe is found

# Load custom-trained model
custom_model_path = "/Users/asya/Downloads/CocktailScanner/cocktail_model.keras"
label_map_path = "/Users/asya/Downloads/CocktailScanner/label_map.txt"

if os.path.exists(custom_model_path) and os.path.exists(label_map_path):
    custom_model = load_model(custom_model_path)
    labels = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
         key, value = line.strip().split(":")
         labels[int(key)] = value
else:
    raise FileNotFoundError("Custom model or label map file not found!")

# Streamlit app
st.title("Cocktail Image Recognition")
st.write("Upload an image of a cocktail to identify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    try:
        img = Image.open(uploaded_file).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize

        # Predict
        predictions = custom_model.predict(img_array)
        confidence = np.max(predictions)
        predicted_label = labels[np.argmax(predictions)]
        
        # Safeguard against encoding issues
        predicted_label = predicted_label.encode ('utf-8').decode('utf-8')

        #Display the result
        st.write(f"The uploaded image is classified as: **{predicted_label}** (Confidence: {confidence:.2f})")

        # Fetch recipe for the predicted cocktail
        recipe, recipe_image = get_cocktail_recipe(predicted_label)
        if recipe:
            st.write(f"**Recipe for {predicted_label}:**")
            st.write(recipe)
            st.image(recipe_image, caption=f"{predicted_label} Image")
        else:
            st.write("No recipe found for the predicted cocktail.")

    except Exception as e:
        st.error(f"Error processing the image: {e}")


