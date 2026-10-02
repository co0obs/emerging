# ==============================================================
# CIFAR-100 FOOD CONTAINER CLASSIFICATION
# PROFESSOR'S BASELINE MODEL (ADAPTED TO FOOD CONTAINERS)
#
# Base Architecture & Pipeline from Professor's Guide:
# - Coarse Class 3: Food Containers (bottle, bowl, can, cup, plate)
# - Single Convolutional Layer (Conv2D 32, 3x3, ReLU)
# - Single MaxPooling2D (2x2)
# - Flatten -> Dense(128, ReLU) -> Dense(5, Softmax)
# - Standard Adam Optimizer (no schedule)
# - Sparse Categorical Crossentropy (no label smoothing)
# - 10 Epochs, validation_split=0.1 (no data augmentation, no dropout)
# ==============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. REPRODUCIBILITY
# ==============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)

print("\n==================================================")
print("PROFESSOR'S BASELINE: FOOD CONTAINER CLASSIFICATION")
print("==================================================")
print("TensorFlow version:", tf.__version__)

COARSE_CLASS = 3  # Food containers

class_names = [
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate"
]

# ==============================================================
# 2. LOAD CIFAR-100 (PROFESSOR'S METHOD)
# ==============================================================

print("\nLoading CIFAR-100...")

# Load coarse and fine labels
(_, train_coarse_labels), (_, test_coarse_labels) = keras.datasets.cifar100.load_data(label_mode="coarse")
(train_images_all, train_fine_labels), (test_images_all, test_fine_labels) = keras.datasets.cifar100.load_data(label_mode="fine")

print("Dataset loaded successfully.")

# ==============================================================
# 3. EXTRACT FOOD CONTAINERS (COARSE CLASS 3)
# ==============================================================

# Extract training samples using professor's indexing logic
train_idx = []
for i in range(len(train_coarse_labels)):
    if train_coarse_labels[i] == COARSE_CLASS:
        train_idx.append(i)
train_idx = np.array(train_idx)

train_images = train_images_all[train_idx]
train_labels = train_fine_labels[train_idx].flatten()

# Extract testing samples using professor's indexing logic
test_idx = []
for i in range(len(test_coarse_labels)):
    if test_coarse_labels[i] == COARSE_CLASS:
        test_idx.append(i)
test_idx = np.array(test_idx)

test_images = test_images_all[test_idx]
test_labels = test_fine_labels[test_idx].flatten()

print(f"\nExtracted Training Images: {len(train_images)}")
print(f"Extracted Testing Images:  {len(test_images)}")

fine_classes = np.unique(train_labels)
print("Original CIFAR-100 fine labels:", fine_classes)

# ==============================================================
# 4. RELABEL CLASSES TO 0-4
# ==============================================================

label_map = {old_label: new_label for new_label, old_label in enumerate(fine_classes)}

train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

print("\nFine-class mapping:")
for new_label, old_label in enumerate(fine_classes):
    print(f"  {old_label} -> {new_label} -> {class_names[new_label]}")

# ==============================================================
# 5. SHOW CLASS DISTRIBUTION
# ==============================================================

class_counts = [np.sum(train_labels == i) for i in range(5)]

plt.figure(num="base_class_distribution", figsize=(8, 5))
plt.bar(class_names, class_counts, color="#3498db", edgecolor="#2980b9")
plt.title("Professor Base: Training Images per Food-Container Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# ==============================================================
# 6. DISPLAY SAMPLE IMAGES
# ==============================================================

plt.figure(num="base_sample_images", figsize=(12, 5))
plot_number = 1

for class_number in range(5):
    class_indexes = np.where(test_labels == class_number)[0]
    for example_number in range(2):
        image_index = class_indexes[example_number]
        plt.subplot(2, 5, plot_number)
        plt.imshow(test_images[image_index])
        plt.title(class_names[class_number])
        plt.axis("off")
        plot_number += 1

plt.suptitle("Sample CIFAR-100 Food Container Images", fontsize=14)
plt.tight_layout()
plt.show()

# ==============================================================
# 7. NORMALIZE IMAGES TO [0, 1]
# ==============================================================

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

print("\nImages normalized to range 0-1.")

# ==============================================================
# 8. BUILD PROFESSOR'S BASELINE CNN
# ==============================================================

print("\nBuilding Professor's Baseline CNN...")

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(5, activation="softmax")
], name="Professor_Base_CNN")

model.summary()

# ==============================================================
# 9. COMPILE MODEL
# ==============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================================================
# 10. TRAIN MODEL (PROFESSOR'S SETTINGS: 10 EPOCHS, SPLIT 0.1)
# ==============================================================

EPOCHS = 10
BATCH_SIZE = 32

print("\n========================================")
print("STARTING TRAINING (10 EPOCHS)")
print("========================================\n")

history = model.fit(
    train_images,
    train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

print("\nTraining finished.")

# ==============================================================
# 11. ACCURACY & LOSS CURVES
# ==============================================================

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
epochs_range = range(1, len(training_accuracy) + 1)

# Accuracy curve
plt.figure(num="base_training_vs_validation_accuracy", figsize=(8, 5))
plt.plot(epochs_range, training_accuracy, label="Training Accuracy", color="#27ae60", marker="o")
plt.plot(epochs_range, validation_accuracy, label="Validation Accuracy", color="#2980b9", marker="o")
plt.title("Professor Base: Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss curve
plt.figure(num="base_training_vs_validation_loss", figsize=(8, 5))
plt.plot(epochs_range, training_loss, label="Training Loss", color="#e67e22", marker="o")
plt.plot(epochs_range, validation_loss, label="Validation Loss", color="#c0392b", marker="o")
plt.title("Professor Base: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================================
# 12. TEST MODEL
# ==============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", round(test_accuracy * 100, 2), "%")

# ==============================================================
# 13. PREDICTIONS
# ==============================================================

predictions = model.predict(test_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# ==============================================================
# 14. CONFUSION MATRIX
# ==============================================================

confusion_matrix = tf.math.confusion_matrix(
    test_labels,
    predicted_labels,
    num_classes=5
).numpy()

plt.figure(num="base_confusion_matrix", figsize=(7, 6))
plt.imshow(confusion_matrix, cmap="Blues")
plt.title("Professor Base: Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(5), class_names, rotation=45)
plt.yticks(range(5), class_names)

for i in range(5):
    for j in range(5):
        plt.text(
            j, i, confusion_matrix[i, j],
            ha="center", va="center",
            color="white" if confusion_matrix[i, j] > np.max(confusion_matrix) / 2 else "black"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# ==============================================================
# 15. PER-CLASS CORRECT CLASSIFICATION RATE
# ==============================================================

class_accuracy = []

print("\n========================================")
print("CORRECT CLASSIFICATION RATE PER CLASS")
print("========================================")

for i in range(5):
    correct = confusion_matrix[i, i]
    total = np.sum(confusion_matrix[i])
    accuracy = correct / total
    class_accuracy.append(accuracy)
    print(f"{class_names[i]} : {round(accuracy * 100, 2)} %")

plt.figure(num="base_per_class_accuracy", figsize=(8, 5))
plt.bar(class_names, np.array(class_accuracy) * 100, color="#16a085", edgecolor="#0e6655")
plt.title("Professor Base: Correct Classification Rate per Class")
plt.xlabel("Food Container")
plt.ylabel("Correctly Classified (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# ==============================================================
# 16. EXAMPLE PREDICTIONS (10 SAMPLES)
# ==============================================================

random_indices = rng.choice(len(test_images), 10, replace=False)

plt.figure(num="base_example_predictions", figsize=(15, 6))
for i, image_index in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[image_index])
    
    pred_c = predicted_labels[image_index]
    true_c = test_labels[image_index]
    conf = predictions[image_index][pred_c] * 100

    plt.title(
        f"Pred: {class_names[pred_c]}\nTrue: {class_names[true_c]}\n{conf:.1f}%",
        fontsize=9
    )
    plt.axis("off")

plt.suptitle("Professor Base: Example Model Predictions", fontsize=14)
plt.tight_layout()
plt.show()

# ==============================================================
# 17. 30 RANDOM TEST PREDICTIONS
# ==============================================================

random_indices_30 = rng.choice(len(test_images), 30, replace=False)

plt.figure(num="base_30_random_test_predictions", figsize=(15, 18))
correct_count = 0

for i, image_index in enumerate(random_indices_30):
    plt.subplot(6, 5, i + 1)
    plt.imshow(test_images[image_index])
    
    true_label = test_labels[image_index]
    pred_label = predicted_labels[image_index]
    conf = predictions[image_index][pred_label] * 100

    if pred_label == true_label:
        res = "CORRECT"
        correct_count += 1
    else:
        res = "WRONG"

    plt.title(
        f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}\n{conf:.1f}%\n{res}",
        fontsize=8,
        color="green" if res == "CORRECT" else "red"
    )
    plt.axis("off")

plt.suptitle("Professor Base: 30 Random Food Container Predictions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

sample_acc = (correct_count / 30) * 100
print(f"\nAccuracy among 30 random images: {round(sample_acc, 2)} %")

# ==============================================================
# 18. MISCLASSIFIED IMAGES
# ==============================================================

wrong_indices = np.where(predicted_labels != test_labels)[0]
print("\nTotal misclassified test images:", len(wrong_indices), f"out of {len(test_images)}")

num_wrong_to_show = min(30, len(wrong_indices))
if num_wrong_to_show > 0:
    selected_wrong = rng.choice(wrong_indices, num_wrong_to_show, replace=False)
    plt.figure(num="base_misclassified_images", figsize=(15, 18))
    for i, img_idx in enumerate(selected_wrong):
        plt.subplot(6, 5, i + 1)
        plt.imshow(test_images[img_idx])
        true_l = test_labels[img_idx]
        pred_l = predicted_labels[img_idx]
        conf = predictions[img_idx][pred_l] * 100
        plt.title(
            f"Pred: {class_names[pred_l]}\nTrue: {class_names[true_l]}\n{conf:.1f}%",
            fontsize=8,
            color="red"
        )
        plt.axis("off")
    plt.suptitle("Professor Base: Misclassified Food Container Images", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# ==============================================================
# 19. FINAL RESULTS
# ==============================================================

best_val_acc = max(validation_accuracy)
best_val_loss = min(validation_loss)

print("\n========================================")
print("FINAL RESULTS (PROFESSOR BASE)")
print("========================================")
print(f"Epochs trained:           {len(training_accuracy)}")
print(f"Final training accuracy:  {round(training_accuracy[-1] * 100, 2)} %")
print(f"Best validation accuracy: {round(best_val_acc * 100, 2)} %")
print(f"Best validation loss:     {round(best_val_loss, 4)}")
print(f"Testing accuracy:         {round(test_accuracy * 100, 2)} %")
print(f"Total misclassified:      {len(wrong_indices)} / {len(test_images)}")
print("========================================")

print("\nProgram finished.")
