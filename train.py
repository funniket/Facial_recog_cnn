# train.py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocessing import load_lfw_funneled_data, clean_and_resize_images, normalize_images, augment_images
from model import create_improved_cnn_model, create_transfer_model

def train_model(use_transfer=False):
    data_dir = r"data"  # Path to your extracted dataset folder
    images, numeric_labels, le = load_lfw_funneled_data(data_dir)
    cleaned_images = clean_and_resize_images(images, target_size=(64, 64))
    normalized_images = normalize_images(cleaned_images)
    if normalized_images.ndim == 3:
        normalized_images = normalized_images[..., None]
    
    # Data augmentation.
    augment_factor = 2
    aug_images = augment_images(normalized_images, augment_factor=augment_factor)
    aug_labels = np.repeat(numeric_labels, repeats=augment_factor)
    
    # Combine original and augmented data.
    X_combined = np.concatenate((normalized_images, aug_images), axis=0)
    y_combined = np.concatenate((numeric_labels, aug_labels), axis=0)
    
    # Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    input_shape = X_train.shape[1:]
    num_classes = len(le.classes_)
    
    if use_transfer:
        model = create_transfer_model(input_shape, num_classes)
    else:
        model = create_improved_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Set up callbacks.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = model.fit(X_train, y_train,
                        epochs=30,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, reduce_lr])
    
    model.save("cnn_face_recognition_model.h5")
    print("Model saved as cnn_face_recognition_model.h5")
    return model, history

if __name__ == '__main__':
    # Set use_transfer=True to use a pretrained MobileNetV2-based model.
    train_model(use_transfer=False)
