# src/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_lfw_funneled_data(data_dir):
    """
    Load images from the LFW funneled folder structure.
    Each subfolder is assumed to be a person (label).
    """
    images = []
    labels = []
    
    for root, dirs, files in os.walk(data_dir):
        # Skip the root folder; process only subdirectories.
        if root == data_dir:
            continue

        person_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    print(f"Warning: Unable to load image: {file_path}")
                    continue
                images.append(img)
                labels.append(person_name)
    images = np.array(images)
    
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    return images, numeric_labels, label_encoder

def clean_and_resize_images(images, target_size=(64, 64)):
    """
    Clean the dataset:
      - Resize images to target_size.
      - Optionally remove outliers based on brightness.
    """
    cleaned_images = []
    for idx, img in enumerate(images):
        # Check for valid image
        if img is None or img.size == 0:
            print(f"Skipping image {idx} due to corruption or missing values.")
            continue
        try:
            # Remove images that are too dark or too bright (simple outlier filtering)
            mean_val = np.mean(img)
            if mean_val < 10 or mean_val > 245:  # thresholds can be adjusted
                print(f"Image {idx} discarded due to extreme brightness/darkness (mean={mean_val}).")
                continue

            resized_img = cv2.resize(img, target_size)
            cleaned_images.append(resized_img)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
    return np.array(cleaned_images)

def normalize_images(images):
    """
    Normalize images to have pixel values between 0 and 1.
    """
    return images.astype('float32') / 255.0

def augment_images(images, augment_factor=2):
    """
    Apply data augmentation to a set of images using Keras's ImageDataGenerator.
    
    Parameters:
        images (numpy.ndarray): Array of images. Expected shape is either (N, height, width) 
                                for grayscale or (N, height, width, channels).
        augment_factor (int): Number of augmented images to generate per original image.
        
    Returns:
        numpy.ndarray: Array containing the augmented images.
    """
    # Define the augmentation configuration.
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    augmented_images = []
    
    # Ensure images have the channel dimension.
    if images.ndim == 3:  # Assuming grayscale images with shape (N, height, width)
        images = images[..., np.newaxis]  # Now shape becomes (N, height, width, 1)
    
    # Loop through each image and generate augmented samples.
    for img in images:
        img = np.expand_dims(img, axis=0)  # Expand to shape (1, height, width, channels)
        aug_iter = datagen.flow(img, batch_size=1)
        for i in range(augment_factor):
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
    
    return np.array(augmented_images)

# Example usage (can be removed or placed under if __name__ == '__main__'):
if __name__ == '__main__':
    data_dir = r"C:\Users\anike\OneDrive\Desktop\Ishant gupta class\Facial_recog_cnn\data\extracted_data\lfw_funneled"
    
    # Load data from directory
    images, numeric_labels, le = load_lfw_funneled_data(data_dir)
    print(f"Loaded {images.shape[0]} images with {len(le.classes_)} unique labels.")
    
    # Clean and resize images
    cleaned_images = clean_and_resize_images(images, target_size=(64, 64))
    print("Cleaned images shape:", cleaned_images.shape)
    
    # Normalize images
    normalized_images = normalize_images(cleaned_images)
    
    # Augment images with an augmentation factor of 2
    aug_images = augment_images(normalized_images, augment_factor=2)
    print("Augmented images shape:", aug_images.shape)
