# ==============================================================
# CIFAR-100 FOOD CONTAINER CLASSIFICATION
# High-Accuracy Engine + Original Output Formatting
# ==============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Dropout,
    SpatialDropout2D,
    Dense,
    GlobalAveragePooling2D
)

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. REPRODUCIBILITY
# ==============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)


# ==============================================================
# 2. BASIC INFORMATION
# ==============================================================

print("\n========================================")
print("CIFAR-100 FOOD CONTAINER CLASSIFICATION")
print("========================================")

print("TensorFlow version:", tf.__version__)

COARSE_CLASS = 3

class_names = [
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate"
]


# ==============================================================
# 3. LOAD CIFAR-100
# ==============================================================

print("\nLoading CIFAR-100...")

(_, train_coarse_labels), (
    _,
    test_coarse_labels
) = keras.datasets.cifar100.load_data(
    label_mode="coarse"
)

(train_images_all, train_fine_labels), (
    test_images_all,
    test_fine_labels
) = keras.datasets.cifar100.load_data(
    label_mode="fine"
)

print("Dataset loaded successfully.")


# ==============================================================
# 4. EXTRACT FOOD CONTAINERS
# ==============================================================

train_indices = np.where(train_coarse_labels.flatten() == COARSE_CLASS)[0]
test_indices = np.where(test_coarse_labels.flatten() == COARSE_CLASS)[0]

food_train_images = train_images_all[train_indices]
food_train_labels_original = train_fine_labels[train_indices].flatten()

food_test_images = test_images_all[test_indices]
food_test_labels_original = test_fine_labels[test_indices].flatten()

# Combine to create proper split
all_images = np.concatenate([food_train_images, food_test_images], axis=0)
all_labels_original = np.concatenate([food_train_labels_original, food_test_labels_original], axis=0)

fine_classes = np.unique(all_labels_original)

print("\nTraining images:", len(food_train_images))
print("Testing images:", len(food_test_images))
print("Original CIFAR-100 fine labels:", fine_classes)


# ==============================================================
# 5. RELABEL CLASSES TO 0-4
# ==============================================================

label_map = {
    old_label: new_label
    for new_label, old_label
    in enumerate(fine_classes)
}

all_labels = np.array([
    label_map[label]
    for label in all_labels_original
])

print("\nFine-class mapping:")

for new_label, old_label in enumerate(fine_classes):
    print(
        old_label,
        "->",
        new_label,
        "->",
        class_names[new_label]
    )

# Create high-accuracy stratified 80:10:10 split
train_indices_final = []
validation_indices_final = []
test_indices_final = []

for class_number in range(5):
    class_indices = np.where(all_labels == class_number)[0]
    rng.shuffle(class_indices)
    
    train_indices_final.extend(class_indices[:480])
    validation_indices_final.extend(class_indices[480:540])
    test_indices_final.extend(class_indices[540:])

rng.shuffle(train_indices_final)
rng.shuffle(validation_indices_final)
rng.shuffle(test_indices_final)

train_images = all_images[train_indices_final]
train_labels = all_labels[train_indices_final]
val_images = all_images[validation_indices_final]
val_labels = all_labels[validation_indices_final]
test_images = all_images[test_indices_final]
test_labels = all_labels[test_indices_final]


# ==============================================================
# 6. SHOW CLASS DISTRIBUTION
# ==============================================================

class_counts = []

for i in range(5):
    class_counts.append(
        np.sum(train_labels == i)
    )

plt.figure(figsize=(8, 5))

plt.bar(
    class_names,
    class_counts
)

plt.title(
    "Training Images per Food-Container Class"
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()


# ==============================================================
# 7. DISPLAY SAMPLE IMAGES
# ==============================================================

plt.figure(figsize=(12, 5))

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

plt.suptitle("Sample CIFAR-100 Food Container Images")
plt.tight_layout()
plt.show()


# ==============================================================
# 8. NORMALIZE
# ==============================================================

train_images = train_images.astype("float32") / 255.0
val_images = val_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Prepare one-hot labels internally for label smoothing to work
train_labels_onehot = keras.utils.to_categorical(train_labels, 5)
val_labels_onehot = keras.utils.to_categorical(val_labels, 5)
test_labels_onehot = keras.utils.to_categorical(test_labels, 5)

print("\nImages normalized to range 0-1.")


# ==============================================================
# 9. DATA AUGMENTATION
# ==============================================================

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.06),
    keras.layers.RandomZoom(0.08),
    keras.layers.RandomTranslation(0.06, 0.06),
    keras.layers.RandomContrast(0.1)
], name="data_augmentation")


# ==============================================================
# 10. BUILD CNN
# ==============================================================

print("\nBuilding CNN...")

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    data_augmentation,

    # Block 1
    Conv2D(64, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Conv2D(64, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Conv2D(64, (3, 3), strides=2, padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    SpatialDropout2D(0.15),

    # Block 2
    Conv2D(128, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Conv2D(128, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Conv2D(128, (3, 3), strides=2, padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    SpatialDropout2D(0.20),

    # Block 3
    Conv2D(256, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Conv2D(256, (3, 3), padding="same", use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    SpatialDropout2D(0.25),

    # Classification Section
    GlobalAveragePooling2D(),
    Dense(256, use_bias=False),
    BatchNormalization(),
    Activation("swish"),
    Dropout(0.35),
    Dense(5, activation="softmax")
])

model.summary()


# ==============================================================
# 11. COMPILE MODEL
# ==============================================================

EPOCHS = 60
BATCH_SIZE = 32
steps_per_epoch = len(train_images) // BATCH_SIZE

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.002,  
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=0.01                    
)

optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.002
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)


# ==============================================================
# 12. CALLBACKS
# ==============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ==============================================================
# 13. TRAIN MODEL
# ==============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")


history = model.fit(
    train_images,
    train_labels_onehot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels_onehot),
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

print("\nTraining finished.")


# ==============================================================
# 14. TRAINING / VALIDATION ACCURACY
# ==============================================================

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

epochs = range(1, len(training_accuracy) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, training_accuracy, label="Training Accuracy")
plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================
# 15. TRAINING / VALIDATION LOSS
# ==============================================================

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(epochs, training_loss, label="Training Loss")
plt.plot(epochs, validation_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================
# 16. TEST MODEL
# ==============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels_onehot,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)


# ==============================================================
# 17. PREDICTIONS
# ==============================================================

predictions = model.predict(test_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)


# ==============================================================
# 18. CONFUSION MATRIX
# ==============================================================

confusion_matrix = tf.math.confusion_matrix(
    test_labels,
    predicted_labels,
    num_classes=5
).numpy()

plt.figure(figsize=(7, 6))
plt.imshow(confusion_matrix)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(5), class_names, rotation=45)
plt.yticks(range(5), class_names)

for i in range(5):
    for j in range(5):
        plt.text(
            j, i, confusion_matrix[i, j],
            ha="center", va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()


# ==============================================================
# 19. PER-CLASS CORRECT CLASSIFICATION RATE
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

    print(
        class_names[i],
        ":",
        round(accuracy * 100, 2),
        "%"
    )

plt.figure(figsize=(8, 5))
plt.bar(class_names, np.array(class_accuracy) * 100)
plt.title("Correct Classification Rate per Class")
plt.xlabel("Food Container")
plt.ylabel("Correctly Classified (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# ==============================================================
# 20. EXAMPLE MODEL PREDICTIONS
# ==============================================================

random_indices = rng.choice(len(test_images), 10, replace=False)

plt.figure(figsize=(15, 6))

for i, image_index in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[image_index])
    
    predicted_class = predicted_labels[image_index]
    true_class = test_labels[image_index]
    confidence = predictions[image_index][predicted_class] * 100

    plt.title(
        f"Pred: {class_names[predicted_class]}\n"
        f"True: {class_names[true_class]}\n"
        f"{confidence:.1f}%",
        fontsize=9
    )
    plt.axis("off")

plt.suptitle("Example Model Predictions")
plt.tight_layout()
plt.show()


# ==============================================================
# 21. 30 RANDOM TEST PREDICTIONS
# ==============================================================

random_indices_30 = rng.choice(len(test_images), 30, replace=False)

plt.figure(figsize=(15, 18))
correct_count = 0

for i, image_index in enumerate(random_indices_30):
    plt.subplot(6, 5, i + 1)
    plt.imshow(test_images[image_index])
    
    true_label = test_labels[image_index]
    predicted_label = predicted_labels[image_index]
    confidence = predictions[image_index][predicted_label] * 100

    if predicted_label == true_label:
        result = "CORRECT"
        correct_count += 1
    else:
        result = "WRONG"

    plt.title(
        f"Pred: {class_names[predicted_label]}\n"
        f"True: {class_names[true_label]}\n"
        f"{confidence:.1f}%\n"
        f"{result}",
        fontsize=8
    )
    plt.axis("off")

plt.suptitle("30 Random CIFAR-100 Food Container Predictions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

sample_accuracy = (correct_count / 30) * 100

print(
    "\nAccuracy among 30 random images:",
    round(sample_accuracy, 2),
    "%"
)


# ==============================================================
# 22. MISCLASSIFIED IMAGES
# ==============================================================

wrong_indices = np.where(predicted_labels != test_labels)[0]

print("\nTotal misclassified test images:", len(wrong_indices))

num_wrong_to_show = min(30, len(wrong_indices))

if num_wrong_to_show > 0:
    selected_wrong_indices = rng.choice(wrong_indices, num_wrong_to_show, replace=False)
    
    plt.figure(figsize=(15, 18))
    
    for i, image_index in enumerate(selected_wrong_indices):
        plt.subplot(6, 5, i + 1)
        plt.imshow(test_images[image_index])
        
        true_label = test_labels[image_index]
        predicted_label = predicted_labels[image_index]
        confidence = predictions[image_index][predicted_label] * 100

        plt.title(
            f"Pred: {class_names[predicted_label]}\n"
            f"True: {class_names[true_label]}\n"
            f"{confidence:.1f}%",
            fontsize=8
        )
        plt.axis("off")

    plt.suptitle("Misclassified Food Container Images", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

else:
    print("No misclassified images.")


# ==============================================================
# 23. FINAL RESULTS
# ==============================================================

best_validation_accuracy = max(validation_accuracy)
best_validation_loss = min(validation_loss)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("Epochs actually trained:", len(training_accuracy))
print("Final training accuracy:", round(training_accuracy[-1] * 100, 2), "%")
print("Best validation accuracy:", round(best_validation_accuracy * 100, 2), "%")
print("Best validation loss:", round(best_validation_loss, 4))
print("Testing accuracy:", round(test_accuracy * 100, 2), "%")

print("\nProgram finished.")