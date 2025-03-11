import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("deepfake_model.h5")

# Function to detect deepfake in an image
def detect_deepfake(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    print("Detection Result:", detect_deepfake(img_path))
