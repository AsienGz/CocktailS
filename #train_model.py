from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Define the path to the dataset
dataset_path = "/Users/asya/Downloads/CocktailScanner/Cocktails/Cocktails"  # Update this to the correct dataset location

# Verify that the dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please verify the path.")

# Image preprocessing and data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize image pixels
    validation_split=0.2  # 20% for validation
)

# Load training data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool features
x = Dense(128, activation="relu")(x)  # Fully connected layer
predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)  # Output layer

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the best model during training
checkpoint_path = "/Users/asya/Downloads/CocktailScanner/cocktail_model.keras"  # Save to the correct location
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Adjust this as needed for better accuracy
    callbacks=[checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save class labels for later use in the Streamlit app
label_map = train_generator.class_indices
label_map = {v: k for k, v in label_map.items()}  # Reverse the dictionary for easy lookup
label_map_path = "/Users/asya/Downloads/CocktailScanner/label_map.txt"
with open(label_map_path, "w") as f:
    for key, value in label_map.items():
        f.write(f"{key}:{value}\n")
print(f"Label map saved at '{label_map_path}'.")




