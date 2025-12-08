import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    ## 1. Enhanced Data Loading and Preprocessing
    def load_and_preprocess_image(image_path, augment=False):
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image {image_path}")
                
            # Convert to RGB and resize with proper interpolation
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_AREA)  # Larger size for EfficientNet
            
            # Data augmentation (only during training)
            if augment:
                # Random rotations
                if np.random.rand() > 0.5:
                    angle = np.random.randint(-45, 45)
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    
                # Random brightness adjustment
                if np.random.rand() > 0.5:
                    img = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, np.random.uniform(-30, 30))
                    
            # Preprocess for EfficientNet
            img = preprocess_input(img)
            
            return img
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    # Load dataset
    parent_dir = Path(r'C:\Users\arsla\Downloads\archive (1)\brain_tumor_dataset')
    filepaths = []
    labels = []

    for label_name in ["yes", "no"]:
        folder_path = parent_dir / label_name
        if folder_path.is_dir():
            for image_file in folder_path.glob("*"):
                if not image_file.name.startswith(".") and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    filepaths.append(image_file)
                    labels.append(label_name)

    # Create DataFrame and shuffle
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    df = shuffle(df, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['filepath'], df['label'], test_size=0.15, stratify=df['label'], random_state=42
    )

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # Modified load_images function to work with Windows multiprocessing
    def load_images(paths, labels, augment=False):
        # Convert to lists for Windows compatibility
        paths_list = paths.tolist() if hasattr(paths, 'tolist') else list(paths)
        labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        
        images = []
        valid_indices = []
        
        for i, path in enumerate(paths_list):
            img = load_and_preprocess_image(path, augment)
            if img is not None:
                images.append(img)
                valid_indices.append(i)
                
        return np.array(images), np.array([labels_list[i] for i in valid_indices])

    print("Loading training images...")
    X_train_images, y_train = load_images(X_train, y_train, augment=True)
    print("Loading validation images...")
    X_val_images, y_val = load_images(X_val, y_val)
    print("Loading test images...")
    X_test_images, y_test = load_images(X_test, y_test)

    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Convert to categorical
    y_train_categorical = to_categorical(y_train_encoded)
    y_val_categorical = to_categorical(y_val_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    ## 2. Enhanced Model Architecture
    def create_model():
        # Use EfficientNetB4 as base with pre-trained weights
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(380, 380, 3),
            pooling=None
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:150]:
            layer.trainable = False
            
        # Unfreeze later layers
        for layer in base_model.layers[150:]:
            layer.trainable = True
            
        # Custom head
        inputs = Input(shape=(380, 380, 3))
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer
        optimizer = Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model

    model = create_model()
    model.summary()

    ## 3. Enhanced Training Configuration
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_brain_tumor_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    # Train the model
    history = model.fit(
        x=X_train_images,
        y=y_train_categorical,
        batch_size=16,
        epochs=50,
        validation_data=(X_val_images, y_val_categorical),
        callbacks=callbacks,
        verbose=1
    )

    ## 4. Evaluation and Testing
    # Load best model
    model = load_model('best_brain_tumor_model.h5')

    # Evaluate on test set
    test_results = model.evaluate(X_test_images, y_test_categorical)
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_classes, target_names=le.classes_))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred_classes))

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

    ## 5. Save Final Model
    model.save('final_brain_tumor_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    freeze_support()
    main()