import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocessing import load_lfw_funneled_data, clean_and_resize_images, normalize_images
from sklearn.model_selection import train_test_split
import numpy as np

def build_model(hp, input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=hp.Int('conv_filters', 32, 128, step=32),
                     kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', 64, 256, step=64), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_tuning():
    data_dir = r"data"
    images, numeric_labels, le = load_lfw_funneled_data(data_dir)
    cleaned_images = clean_and_resize_images(images, target_size=(64, 64))
    normalized_images = normalize_images(cleaned_images)
    if normalized_images.ndim == 3:
        normalized_images = normalized_images[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(normalized_images, numeric_labels, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    num_classes = len(le.classes_)
    
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='face_recognition_tuning'
    )
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:", best_hyperparameters.values)
    return best_model, best_hyperparameters

if __name__ == '__main__':
    run_tuning()
