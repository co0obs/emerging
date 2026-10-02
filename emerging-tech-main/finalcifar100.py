# ==============================================================
# CIFAR-100 FOOD CONTAINER CLASSIFICATION
# FINAL MODEL
#
# Classes:
# 0 = bottle
# 1 = bowl
# 2 = can
# 3 = cup
# 4 = plate
#
# Dataset split:
# 80% Training
# 10% Validation
# 10% Testing
# ==============================================================

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Dropout,
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

print("\n==========================================")
print("CIFAR-100 FOOD CONTAINER CLASSIFICATION")
print("FINAL CNN MODEL")
print("==========================================")

print("TensorFlow version:", tf.__version__)


COARSE_CLASS = 3

class_names = [
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate"
]

NUM_CLASSES = 5


# ==============================================================
# 3. LOAD CIFAR-100
# ==============================================================

print("\nLoading CIFAR-100...")


# Load coarse labels
(train_images_coarse, train_coarse_labels), (
    test_images_coarse,
    test_coarse_labels
) = keras.datasets.cifar100.load_data(
    label_mode="coarse"
)


# Load fine labels
(train_images_fine, train_fine_labels), (
    test_images_fine,
    test_fine_labels
) = keras.datasets.cifar100.load_data(
    label_mode="fine"
)


print("CIFAR-100 loaded successfully.")


# ==============================================================
# 4. EXTRACT FOOD-CONTAINER IMAGES
# ==============================================================

train_indices = np.where(
    train_coarse_labels.flatten() == COARSE_CLASS
)[0]

test_indices = np.where(
    test_coarse_labels.flatten() == COARSE_CLASS
)[0]


food_train_images = train_images_fine[
    train_indices
]

food_train_labels_original = train_fine_labels[
    train_indices
].flatten()


food_test_images = test_images_fine[
    test_indices
]

food_test_labels_original = test_fine_labels[
    test_indices
].flatten()


print(
    "\nOriginal CIFAR food-container training images:",
    len(food_train_images)
)

print(
    "Original CIFAR food-container testing images:",
    len(food_test_images)
)


# ==============================================================
# 5. COMBINE ALL FOOD-CONTAINER IMAGES
#
# We have:
# 2500 original training images
# 500 original testing images
#
# Total = 3000 images
#
# We will make our own stratified:
# 80% / 10% / 10% split
# ==============================================================

all_images = np.concatenate(
    [
        food_train_images,
        food_test_images
    ],
    axis=0
)


all_labels_original = np.concatenate(
    [
        food_train_labels_original,
        food_test_labels_original
    ],
    axis=0
)


print(
    "\nTotal food-container images:",
    len(all_images)
)


# ==============================================================
# 6. RELABEL CIFAR FINE CLASSES TO 0-4
# ==============================================================

fine_classes = np.unique(
    all_labels_original
)


label_map = {
    original_label: new_label

    for new_label, original_label
    in enumerate(fine_classes)
}


all_labels = np.array([
    label_map[label]

    for label
    in all_labels_original
])


print("\nFine-class mapping:")


for new_label, original_label in enumerate(
    fine_classes
):

    print(
        original_label,
        "->",
        new_label,
        "->",
        class_names[new_label]
    )


# ==============================================================
# 7. STRATIFIED 80:10:10 SPLIT
#
# Each class has 600 total images.
#
# Per class:
# 480 training
# 60 validation
# 60 testing
#
# Total:
# 2400 training
# 300 validation
# 300 testing
# ==============================================================

train_indices_final = []
validation_indices_final = []
test_indices_final = []


for class_number in range(NUM_CLASSES):

    class_indices = np.where(
        all_labels == class_number
    )[0]


    # Shuffle each class independently
    rng.shuffle(
        class_indices
    )


    total_class_images = len(
        class_indices
    )


    train_end = int(
        total_class_images * 0.80
    )


    validation_end = int(
        total_class_images * 0.90
    )


    train_indices_final.extend(
        class_indices[:train_end]
    )


    validation_indices_final.extend(
        class_indices[
            train_end:validation_end
        ]
    )


    test_indices_final.extend(
        class_indices[
            validation_end:
        ]
    )


train_indices_final = np.array(
    train_indices_final
)

validation_indices_final = np.array(
    validation_indices_final
)

test_indices_final = np.array(
    test_indices_final
)


# Shuffle after combining classes
rng.shuffle(
    train_indices_final
)

rng.shuffle(
    validation_indices_final
)

rng.shuffle(
    test_indices_final
)


# Create final datasets
train_images = all_images[
    train_indices_final
]

train_labels = all_labels[
    train_indices_final
]


validation_images = all_images[
    validation_indices_final
]

validation_labels = all_labels[
    validation_indices_final
]


test_images = all_images[
    test_indices_final
]

test_labels = all_labels[
    test_indices_final
]


print("\n==========================================")
print("FINAL DATASET SPLIT")
print("==========================================")

print(
    "Training:",
    len(train_images),
    "images"
)

print(
    "Validation:",
    len(validation_images),
    "images"
)

print(
    "Testing:",
    len(test_images),
    "images"
)


print("\nImages per class:")


for i in range(NUM_CLASSES):

    print(
        class_names[i],
        "| Train:",
        np.sum(train_labels == i),
        "| Validation:",
        np.sum(validation_labels == i),
        "| Test:",
        np.sum(test_labels == i)
    )


# ==============================================================
# 8. CLASS DISTRIBUTION GRAPH
# ==============================================================

training_class_counts = [
    np.sum(
        train_labels == i
    )

    for i in range(NUM_CLASSES)
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    class_names,
    training_class_counts
)

plt.title(
    "Training Images per Food-Container Class"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Images"
)

plt.tight_layout()

plt.savefig(
    "01_training_class_distribution.png",
    dpi=200
)

plt.show()


# ==============================================================
# 9. DISPLAY SAMPLE IMAGES
# ==============================================================

plt.figure(
    figsize=(12, 5)
)


plot_number = 1


for class_number in range(NUM_CLASSES):

    class_indexes = np.where(
        test_labels == class_number
    )[0]


    for example_number in range(2):

        image_index = class_indexes[
            example_number
        ]


        plt.subplot(
            2,
            5,
            plot_number
        )


        plt.imshow(
            test_images[
                image_index
            ]
        )


        plt.title(
            class_names[
                class_number
            ]
        )


        plt.axis(
            "off"
        )


        plot_number += 1


plt.suptitle(
    "Sample CIFAR-100 Food Container Images"
)

plt.tight_layout()

plt.savefig(
    "02_sample_images.png",
    dpi=200
)

plt.show()


# ==============================================================
# 10. NORMALIZE PIXEL VALUES
#
# Original:
# 0 - 255
#
# Normalized:
# 0.0 - 1.0
# ==============================================================

train_images = (
    train_images.astype("float32")
    / 255.0
)

validation_images = (
    validation_images.astype("float32")
    / 255.0
)

test_images = (
    test_images.astype("float32")
    / 255.0
)


print(
    "\nImages normalized from 0-255 to 0-1."
)


# ==============================================================
# 11. ONE-HOT ENCODE LABELS
#
# Required because we are using
# categorical crossentropy with label smoothing.
# ==============================================================

train_labels_onehot = keras.utils.to_categorical(
    train_labels,
    NUM_CLASSES
)

validation_labels_onehot = keras.utils.to_categorical(
    validation_labels,
    NUM_CLASSES
)

test_labels_onehot = keras.utils.to_categorical(
    test_labels,
    NUM_CLASSES
)


# ==============================================================
# 12. DATA AUGMENTATION
#
# Mild augmentation because CIFAR images
# are only 32 x 32 pixels.
# ==============================================================

data_augmentation = keras.Sequential(
    [

        keras.layers.RandomFlip(
            "horizontal"
        ),

        keras.layers.RandomRotation(
            0.04
        ),

        keras.layers.RandomZoom(
            0.08
        ),

        keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )

    ],

    name="data_augmentation"
)


# ==============================================================
# 13. BUILD CNN
# ==============================================================

print(
    "\nBuilding final CNN..."
)


model = keras.Sequential([


    # ==========================================================
    # INPUT
    # ==========================================================

    keras.Input(
        shape=(32, 32, 3)
    ),


    # ==========================================================
    # AUGMENTATION
    # ==========================================================

    data_augmentation,


    # ==========================================================
    # CONVOLUTION BLOCK 1
    #
    # 32 filters
    # ==========================================================

    Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    MaxPooling2D(
        pool_size=(2, 2)
    ),


    Dropout(
        0.20
    ),


    # ==========================================================
    # CONVOLUTION BLOCK 2
    #
    # 64 filters
    # ==========================================================

    Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    MaxPooling2D(
        pool_size=(2, 2)
    ),


    Dropout(
        0.25
    ),


    # ==========================================================
    # CONVOLUTION BLOCK 3
    #
    # 128 filters
    # ==========================================================

    Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    MaxPooling2D(
        pool_size=(2, 2)
    ),


    Dropout(
        0.30
    ),


    # ==========================================================
    # GLOBAL FEATURE EXTRACTION
    #
    # Replaces Flatten()
    # ==========================================================

    GlobalAveragePooling2D(),


    # ==========================================================
    # DENSE CLASSIFIER
    # ==========================================================

    Dense(
        128,
        use_bias=False
    ),

    BatchNormalization(),

    Activation(
        "relu"
    ),


    Dropout(
        0.40
    ),


    # ==========================================================
    # OUTPUT
    #
    # 5 output probabilities
    # ==========================================================

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# Display architecture
model.summary()


# ==============================================================
# 14. OPTIMIZER
#
# AdamW = Adam + weight decay
# ==============================================================

optimizer = keras.optimizers.AdamW(

    learning_rate=0.001,

    weight_decay=0.0001

)


# ==============================================================
# 15. LOSS FUNCTION
#
# Label smoothing helps reduce overconfidence.
# ==============================================================

loss_function = keras.losses.CategoricalCrossentropy(

    label_smoothing=0.05

)


# ==============================================================
# 16. COMPILE MODEL
# ==============================================================

model.compile(

    optimizer=optimizer,

    loss=loss_function,

    metrics=[
        "accuracy"
    ]

)


# ==============================================================
# 17. CALLBACKS
# ==============================================================

early_stopping = keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1

)


reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.000001,

    verbose=1

)


# ==============================================================
# 18. TRAIN MODEL
#
# 50 = MAXIMUM epochs
#
# Early stopping can stop it sooner.
# ==============================================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================\n")


history = model.fit(

    train_images,

    train_labels_onehot,

    validation_data=(

        validation_images,

        validation_labels_onehot

    ),

    epochs=50,

    batch_size=32,

    callbacks=[

        early_stopping,

        reduce_learning_rate

    ],

    shuffle=True,

    verbose=1

)


print(
    "\nTraining completed."
)


# ==============================================================
# 19. TRAINING ACCURACY GRAPH
# ==============================================================

training_accuracy = history.history[
    "accuracy"
]

validation_accuracy = history.history[
    "val_accuracy"
]


epoch_numbers = range(

    1,

    len(training_accuracy) + 1

)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    epoch_numbers,

    training_accuracy,

    label="Training Accuracy"

)


plt.plot(

    epoch_numbers,

    validation_accuracy,

    label="Validation Accuracy"

)


plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "03_training_validation_accuracy.png",
    dpi=200
)


plt.show()


# ==============================================================
# 20. TRAINING LOSS GRAPH
# ==============================================================

training_loss = history.history[
    "loss"
]

validation_loss = history.history[
    "val_loss"
]


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    epoch_numbers,

    training_loss,

    label="Training Loss"

)


plt.plot(

    epoch_numbers,

    validation_loss,

    label="Validation Loss"

)


plt.title(
    "Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "04_training_validation_loss.png",
    dpi=200
)


plt.show()


# ==============================================================
# 21. TEST MODEL
# ==============================================================

print("\n==========================================")
print("TESTING MODEL")
print("==========================================")


test_loss, test_accuracy = model.evaluate(

    test_images,

    test_labels_onehot,

    verbose=1

)


print(
    "\nTest Loss:",
    test_loss
)


print(
    "Test Accuracy:",
    test_accuracy
)


print(
    "Test Accuracy (%):",
    test_accuracy * 100
)


# ==============================================================
# 22. MAKE PREDICTIONS
# ==============================================================

predictions = model.predict(

    test_images,

    verbose=0

)


predicted_labels = np.argmax(

    predictions,

    axis=1

)


# ==============================================================
# 23. CONFUSION MATRIX
# ==============================================================

confusion_matrix = tf.math.confusion_matrix(

    test_labels,

    predicted_labels,

    num_classes=NUM_CLASSES

).numpy()


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    confusion_matrix
)


plt.title(
    "Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "True Class"
)


plt.xticks(

    range(NUM_CLASSES),

    class_names,

    rotation=45

)


plt.yticks(

    range(NUM_CLASSES),

    class_names

)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            confusion_matrix[i, j],

            ha="center",

            va="center"

        )


plt.colorbar()

plt.tight_layout()


plt.savefig(
    "05_confusion_matrix.png",
    dpi=200
)


plt.show()


# ==============================================================
# 24. CORRECT CLASSIFICATION RATE PER CLASS
# ==============================================================

class_accuracy = []


print("\n==========================================")
print("CORRECT CLASSIFICATION RATE PER CLASS")
print("==========================================")


for i in range(NUM_CLASSES):


    correct = confusion_matrix[
        i,
        i
    ]


    total = np.sum(
        confusion_matrix[i]
    )


    accuracy = (
        correct / total
    )


    class_accuracy.append(
        accuracy
    )


    print(

        class_names[i],

        ":",

        round(
            accuracy * 100,
            2
        ),

        "%"

    )


plt.figure(
    figsize=(8, 5)
)


plt.bar(

    class_names,

    np.array(
        class_accuracy
    ) * 100

)


plt.title(
    "Correct Classification Rate per Class"
)


plt.xlabel(
    "Food Container"
)


plt.ylabel(
    "Correctly Classified (%)"
)


plt.ylim(
    0,
    100
)


plt.tight_layout()


plt.savefig(
    "06_class_accuracy.png",
    dpi=200
)


plt.show()


# ==============================================================
# 25. 10 EXAMPLE PREDICTIONS
# ==============================================================

example_indices = rng.choice(

    len(test_images),

    10,

    replace=False

)


plt.figure(
    figsize=(15, 6)
)


for i, image_index in enumerate(
    example_indices
):


    plt.subplot(
        2,
        5,
        i + 1
    )


    plt.imshow(
        test_images[
            image_index
        ]
    )


    predicted_class = predicted_labels[
        image_index
    ]


    true_class = test_labels[
        image_index
    ]


    confidence = (

        predictions[
            image_index
        ][
            predicted_class
        ]

        * 100

    )


    if predicted_class == true_class:

        result = "CORRECT"

    else:

        result = "WRONG"


    plt.title(

        f"Pred: {class_names[predicted_class]}\n"
        f"True: {class_names[true_class]}\n"
        f"{confidence:.1f}% - {result}",

        fontsize=8

    )


    plt.axis(
        "off"
    )


plt.suptitle(
    "Example Model Predictions"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


plt.savefig(
    "07_example_predictions.png",
    dpi=200
)


plt.show()


# ==============================================================
# 26. 30 RANDOM TEST PREDICTIONS
# ==============================================================

random_indices = rng.choice(

    len(test_images),

    30,

    replace=False

)


correct_count = 0


plt.figure(
    figsize=(15, 18)
)


for i, image_index in enumerate(
    random_indices
):


    plt.subplot(
        6,
        5,
        i + 1
    )


    plt.imshow(
        test_images[
            image_index
        ]
    )


    true_class = test_labels[
        image_index
    ]


    predicted_class = predicted_labels[
        image_index
    ]


    confidence = (

        predictions[
            image_index
        ][
            predicted_class
        ]

        * 100

    )


    if predicted_class == true_class:

        result = "CORRECT"

        correct_count += 1

    else:

        result = "WRONG"


    plt.title(

        f"Pred: {class_names[predicted_class]}\n"
        f"True: {class_names[true_class]}\n"
        f"{confidence:.1f}%\n"
        f"{result}",

        fontsize=8

    )


    plt.axis(
        "off"
    )


plt.suptitle(

    "30 Random CIFAR-100 Food Container Predictions",

    fontsize=16

)


plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)


plt.savefig(
    "08_30_random_predictions.png",
    dpi=200
)


plt.show()


sample_accuracy = (

    correct_count

    / 30

    * 100

)


print(

    "\nAccuracy among 30 random test images:",

    round(
        sample_accuracy,
        2
    ),

    "%"

)


# ==============================================================
# 27. DISPLAY 30 MISCLASSIFIED IMAGES
# ==============================================================

wrong_indices = np.where(

    predicted_labels != test_labels

)[0]


print(

    "\nTotal misclassified test images:",

    len(
        wrong_indices
    )

)


num_wrong_to_show = min(

    30,

    len(
        wrong_indices
    )

)


if num_wrong_to_show > 0:


    selected_wrong = rng.choice(

        wrong_indices,

        num_wrong_to_show,

        replace=False

    )


    plt.figure(
        figsize=(15, 18)
    )


    for i, image_index in enumerate(
        selected_wrong
    ):


        plt.subplot(
            6,
            5,
            i + 1
        )


        plt.imshow(
            test_images[
                image_index
            ]
        )


        true_class = test_labels[
            image_index
        ]


        predicted_class = predicted_labels[
            image_index
        ]


        confidence = (

            predictions[
                image_index
            ][
                predicted_class
            ]

            * 100

        )


        plt.title(

            f"Pred: {class_names[predicted_class]}\n"
            f"True: {class_names[true_class]}\n"
            f"{confidence:.1f}%",

            fontsize=8

        )


        plt.axis(
            "off"
        )


    plt.suptitle(

        "Misclassified Food Container Images",

        fontsize=16

    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )


    plt.savefig(
        "09_misclassified_images.png",
        dpi=200
    )


    plt.show()


# ==============================================================
# 28. FINAL STATISTICS
# ==============================================================

best_validation_accuracy = max(
    validation_accuracy
)


best_validation_accuracy_epoch = (

    np.argmax(
        validation_accuracy
    )

    + 1

)


best_validation_loss = min(
    validation_loss
)


best_validation_loss_epoch = (

    np.argmin(
        validation_loss
    )

    + 1

)


print("\n==========================================")
print("FINAL RESULTS")
print("==========================================")


print(
    "Dataset split:"
)


print(
    "Training:",
    len(train_images)
)


print(
    "Validation:",
    len(validation_images)
)


print(
    "Testing:",
    len(test_images)
)


print(
    "\nEpochs actually trained:",
    len(training_accuracy)
)


print(
    "Final training accuracy:",
    round(
        training_accuracy[-1]
        * 100,
        2
    ),
    "%"
)


print(
    "Best validation accuracy:",
    round(
        best_validation_accuracy
        * 100,
        2
    ),
    "%"
)


print(
    "Best validation accuracy epoch:",
    best_validation_accuracy_epoch
)


print(
    "Best validation loss:",
    round(
        best_validation_loss,
        4
    )
)


print(
    "Best validation loss epoch:",
    best_validation_loss_epoch
)


print(
    "Final test accuracy:",
    round(
        test_accuracy
        * 100,
        2
    ),
    "%"
)


print(
    "Final test loss:",
    round(
        test_loss,
        4
    )
)


# ==============================================================
# 29. SAVE MODEL
# ==============================================================

model.save(
    "foodcontainers_cifar100_final.keras"
)


print(
    "\nModel saved as:"
)

print(
    "foodcontainers_cifar100_final.keras"
)


print(
    "\nAll major graphs were also saved as PNG files."
)


print(
    "\nProgram finished."
)