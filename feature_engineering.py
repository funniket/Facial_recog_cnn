# feature_engineering.py
import numpy as np
from tfds_preprocessing import load_lfw_funneled_data, clean_and_resize_images, normalize_images, augment_images

data_dir = r"data"
images, numeric_labels, le = load_lfw_funneled_data(data_dir)
cleaned_images = clean_and_resize_images(images, target_size=(64, 64))
normalized_images = normalize_images(cleaned_images)
if normalized_images.ndim == 3:
    normalized_images = normalized_images[..., None]

# Augment images.
augment_factor = 2
aug_images = augment_images(normalized_images, augment_factor=augment_factor)
# Replicate labels for augmented images.
aug_labels = np.repeat(numeric_labels, repeats=augment_factor)

# Combine original and augmented images.
X_combined = np.concatenate((normalized_images, aug_images), axis=0)
y_combined = np.concatenate((numeric_labels, aug_labels), axis=0)
print("Combined images shape:", X_combined.shape)
print("Combined labels shape:", y_combined.shape)
