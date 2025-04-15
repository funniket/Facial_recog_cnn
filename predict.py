import numpy as np
import cv2
import tensorflow as tf
from preprocessing import load_lfw_funneled_data  # Ensure preprocessing.py is in the same folder
# No pickle saving and no manual mapping: we recreate the mapping by reloading the data.

def preprocess_image_file(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to load image at {image_path}")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dimension
    img = np.expand_dims(img, axis=0)   # add batch dimension
    return img

def predict_image(image_path, model_path="cnn_face_recognition_model.h5", data_dir=r"data"):
    # Load the trained model.
    model = tf.keras.models.load_model(model_path)
    processed_img = preprocess_image_file(image_path)
    pred_probs = model.predict(processed_img)
    predicted_numeric = int(np.argmax(pred_probs, axis=1)[0])
    
    # Recompute the mapping by loading the dataset.
    # This assumes that the directory structure in `data_dir` is the same as during training.
    _, _, le = load_lfw_funneled_data(data_dir)
    predicted_class = le.inverse_transform([predicted_numeric])[0]
    
    return predicted_class

if __name__ == '__main__':
    test_image_path = "C:/Users/anike/OneDrive/Desktop/Ishant gupta class/Facial_recog_cnn/leo.jpg"  # Replace with your image path.
    prediction = predict_image(test_image_path)
    print("Predicted class:", prediction)
