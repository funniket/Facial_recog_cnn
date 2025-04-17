import streamlit as st
import numpy as np
# import cv2
try:
    import cv2
except ImportError:
    !pip install opencv-python
    import cv2
import tensorflow as tf
from PIL import Image
import os

# Set app title and description
st.set_page_config(page_title="Face Recognition System", page_icon=":camera:")
st.title("Face Recognition System")
st.write("Upload an image of a face to recognize the person")

# Sidebar for additional options
with st.sidebar:
    st.header("About")
    st.write("This app uses a CNN model trained on the LFW dataset to recognize faces.")
    st.write("The model was trained using TensorFlow and deployed with Streamlit.")
    
    st.header("Model Information")
    st.write("- Input size: 64x64 grayscale")
    st.write("- Architecture: Custom CNN with data augmentation")
    
    st.header("Instructions")
    st.write("1. Upload a clear face image (jpg, png)")
    st.write("2. The app will process and display the prediction")
    st.write("3. Try with different angles/lighting for best results")

# Load the model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("cnn_face_recognition_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to preprocess an uploaded image
def preprocess_image(image, target_size=(64, 64)):
    try:
        # Convert to numpy array
        img = np.array(image)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # add channel dimension
        img = np.expand_dims(img, axis=0)   # add batch dimension
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to get class names by scanning the data directory
@st.cache_data
def get_class_names(data_dir="data"):
    try:
        class_names = []
        for root, dirs, files in os.walk(data_dir):
            if root != data_dir:  # Skip the root directory
                person_name = os.path.basename(root)
                class_names.append(person_name)
        return sorted(list(set(class_names)))
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return []

class_names = get_class_names()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_img = preprocess_image(image)
    
    if processed_img is not None and model is not None:
        with st.spinner("Analyzing the image..."):
            # Make prediction
            pred_probs = model.predict(processed_img)
            predicted_idx = int(np.argmax(pred_probs, axis=1)[0])
            confidence = float(np.max(pred_probs, axis=1)[0])
            
            # Display results
            st.subheader("Prediction Results")
            
            if len(class_names) > 0 and predicted_idx < len(class_names):
                predicted_class = class_names[predicted_idx]
                st.success(f"Predicted: **{predicted_class}** (Confidence: {confidence:.2%})")
            else:
                st.warning(f"Predicted class index {predicted_idx} (Confidence: {confidence:.2%})")
                st.info("Could not map index to class name - check your data directory")
            
            # Show confidence distribution
            st.write("Confidence distribution across classes:")
            st.bar_chart(pred_probs[0])
            
            # Show top 5 predictions if we have class names
            if len(class_names) > 0:
                top_k = 5
                top_indices = np.argsort(pred_probs[0])[-top_k:][::-1]
                st.write("Top predictions:")
                for i, idx in enumerate(top_indices):
                    if idx < len(class_names):
                        st.write(f"{i+1}. {class_names[idx]} ({pred_probs[0][idx]:.2%})")
                    else:
                        st.write(f"{i+1}. Class index {idx} ({pred_probs[0][idx]:.2%})")