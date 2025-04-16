# exploration.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tfds_preprocessing import load_lfw_funneled_data, clean_and_resize_images, normalize_images

# Set the path to your extracted data.
data_dir = r"data"
images, numeric_labels, le = load_lfw_funneled_data(data_dir)
cleaned_images = clean_and_resize_images(images, target_size=(64, 64))
normalized_images = normalize_images(cleaned_images)
if normalized_images.ndim == 3:
    normalized_images = normalized_images[..., None]

# Visualize a set of random sample images.
sample_indices = np.random.choice(range(normalized_images.shape[0]), 6, replace=False)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    axes[i].imshow(normalized_images[idx].squeeze(), cmap="gray")
    axes[i].set_title(f"Label: {numeric_labels[idx]}")
    axes[i].axis("off")
plt.suptitle("Sample Normalized Images")
plt.show()

# Plot class distribution.
df = pd.DataFrame({'label': numeric_labels})
plt.figure(figsize=(10, 4))
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.xlabel("Class (encoded)")
plt.ylabel("Count")
plt.show()
