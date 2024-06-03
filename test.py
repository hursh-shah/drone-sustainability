import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the SavedModel
model_path = './1'
model = tf.saved_model.load(model_path)
print(loaded_model)
"""
print("Model Input Shape:", loaded_model.input_shape)

def predict(model, img_path):
    img = load_img(img_path, target_size=(your_target_size))  # Set your_target_size to the expected input size of your model
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


# Specify the path to the individual image
img_path = 'grass.jpeg'

# Use the predict function
predicted_class, confidence = predict(loaded_model, img_path)

# Print or use the predicted class and confidence
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")

# Display the image
img = plt.imread(img_path)
plt.imshow(img)
plt.title(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence}%")
plt.axis("off")
plt.show()
"""