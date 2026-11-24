
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# STEP 1: Define dataset
DATASET_PATH = "dataset"
CATEGORIES = ["Apple", "Banana", "Orange"]
IMG_SIZE = 64

features = []
labels = []

# STEP 2: Load and preprocess images
for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                image = image.resize((IMG_SIZE, IMG_SIZE))

                image_array = np.asarray(
                    image, dtype=np.float32
                ) / 255.0

                features.append(image_array.flatten())
                labels.append(label)

        except (OSError, ValueError):
            print("Skipping invalid image:", filename)

X = np.array(features)
y = np.array(labels)

print("Total images loaded:", len(X))

# STEP 3: Validate dataset
for label, category in enumerate(CATEGORIES):
    count = np.sum(y == label)
    print(f"{category}: {count} images")

    if count < 5:
        raise ValueError(
            f"Add at least 5 images to {category}"
        )

# STEP 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 5: Create KNN model
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95, svd_solver="full"),
    KNeighborsClassifier(n_neighbors=3)
)

# STEP 6: Train model
print("\nTraining KNN model...")
model.fit(X_train, y_train)

# STEP 7: Test model
predictions = model.predict(X_test)

# STEP 8: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CATEGORIES,
        zero_division=0
    )
)

# STEP 9: Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1, 2],
    display_labels=CATEGORIES,
    cmap="Blues"
)

plt.title("FruitVision - KNN Confusion Matrix")
plt.tight_layout()
plt.show()

# STEP 10: Save trained model
joblib.dump(model, "fruit_knn_model.pkl")
print("Model saved successfully!")

# STEP 11: Predict a new image
test_path = "test_images/test_fruit.jpg"

if os.path.isfile(test_path):
    with Image.open(test_path) as img:
        rgb_image = img.convert("RGB")
        resized = rgb_image.resize((IMG_SIZE, IMG_SIZE))

        vector = (
            np.asarray(resized, dtype=np.float32) / 255.0
        ).flatten().reshape(1, -1)

    result = model.predict(vector)[0]

    print("\nPredicted Fruit:", CATEGORIES[result])

    plt.imshow(rgb_image)
    plt.title(f"Prediction: {CATEGORIES[result]}")
    plt.axis("off")
    plt.show()

else:
    print("\nNo test image found.")
    print("Add test_fruit.jpg inside test_images/")
