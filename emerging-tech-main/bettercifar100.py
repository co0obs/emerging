# ==============================================================
# CIFAR-100 FOOD CONTAINER CLASSIFICATION
# IMPROVED CNN EXPERIMENT
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

from tensorflow.keras import regularizers

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
# 2. BASIC SETTINGS
# ==============================================================

print("\n========================================")
print("CIFAR-100 FOOD CONTAINER CLASSIFICATION")
print("IMPROVED CNN")
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


print("Dataset loaded.")


# ==============================================================
# 4. EXTRACT FOOD CONTAINER SUPERCLASS
# ==============================================================

train_indices = np.where(
    train_coarse_labels.flatten() == COARSE_CLASS
)[0]


test_indices = np.where(
    test_coarse_labels.flatten() == COARSE_CLASS
)[0]


all_train_images = train_images_all[
    train_indices
]


original_train_labels = train_fine_labels[
    train_indices
].flatten()


test_images = test_images_all[
    test_indices
]


original_test_labels = test_fine_labels[
    test_indices
].flatten()


print(
    "\nFood-container training images:",
    len(all_train_images)
)

print(
    "Food-container testing images:",
    len(test_images)
)


# ==============================================================
# 5. MAP ORIGINAL CIFAR LABELS TO 0-4
# ==============================================================

fine_classes = np.unique(
    original_train_labels
)


label_map = {
    old_label: new_label
    for new_label, old_label
    in enumerate(fine_classes)
}


all_train_labels = np.array([
    label_map[label]
    for label in original_train_labels
])


test_labels = np.array([
    label_map[label]
    for label in original_test_labels
])


print("\nLabel mapping:")

for new_label, old_label in enumerate(
    fine_classes
):

    print(
        old_label,
        "->",
        new_label,
        "->",
        class_names[new_label]
    )


# ==============================================================
# 6. CLASS DISTRIBUTION
# ==============================================================

class_counts = []

for i in range(5):

    class_counts.append(
        np.sum(all_train_labels == i)
    )


plt.figure(
    figsize=(8, 5)
)

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
# 7. SAMPLE CIFAR-100 IMAGES
# ==============================================================

plt.figure(
    figsize=(12, 5)
)

plot_number = 1


for class_number in range(5):

    indexes = np.where(
        test_labels == class_number
    )[0]


    for j in range(2):

        image_index = indexes[j]

        plt.subplot(
            2,
            5,
            plot_number
        )

        plt.imshow(
            test_images[image_index]
        )

        plt.title(
            class_names[class_number]
        )

        plt.axis("off")

        plot_number += 1


plt.suptitle(
    "Sample CIFAR-100 Food Container Images"
)

plt.tight_layout()
plt.show()


# ==============================================================
# 8. NORMALIZE IMAGES
# ==============================================================

all_train_images = (
    all_train_images.astype("float32")
    / 255.0
)


test_images = (
    test_images.astype("float32")
    / 255.0
)


# ==============================================================
# 9. STRATIFIED TRAIN / VALIDATION SPLIT
# ==============================================================

# 80% training
# 20% validation
#
# Done separately for every class so every class
# has equal representation.


train_split_indices = []
validation_split_indices = []


for class_number in range(5):

    indexes = np.where(
        all_train_labels == class_number
    )[0]


    rng.shuffle(
        indexes
    )


    split_point = int(
        len(indexes) * 0.80
    )


    train_split_indices.extend(
        indexes[:split_point]
    )


    validation_split_indices.extend(
        indexes[split_point:]
    )


train_split_indices = np.array(
    train_split_indices
)


validation_split_indices = np.array(
    validation_split_indices
)


rng.shuffle(
    train_split_indices
)


rng.shuffle(
    validation_split_indices
)


train_images = all_train_images[
    train_split_indices
]


train_labels = all_train_labels[
    train_split_indices
]


validation_images = all_train_images[
    validation_split_indices
]


validation_labels = all_train_labels[
    validation_split_indices
]


print("\n========================================")
print("DATA SPLIT")
print("========================================")

print(
    "Training images:",
    len(train_images)
)

print(
    "Validation images:",
    len(validation_images)
)

print(
    "Testing images:",
    len(test_images)
)


# ==============================================================
# 10. DATA AUGMENTATION
# ==============================================================

data_augmentation = keras.Sequential([

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
    ),

    keras.layers.RandomContrast(
        0.08
    )

], name="augmentation")


# ==============================================================
# 11. BUILD IMPROVED CNN
# ==============================================================

print("\nBuilding improved CNN...")


model = keras.Sequential([

    keras.Input(
        shape=(32, 32, 3)
    ),


    # ----------------------------------------------------------
    # DATA AUGMENTATION
    # ----------------------------------------------------------

    data_augmentation,


    # ----------------------------------------------------------
    # CONVOLUTION BLOCK 1
    # ----------------------------------------------------------

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
        (2, 2)
    ),

    Dropout(
        0.20
    ),


    # ----------------------------------------------------------
    # CONVOLUTION BLOCK 2
    # ----------------------------------------------------------

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
        (2, 2)
    ),

    Dropout(
        0.25
    ),


    # ----------------------------------------------------------
    # CONVOLUTION BLOCK 3
    # ----------------------------------------------------------

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
        (2, 2)
    ),

    Dropout(
        0.30
    ),


    # ----------------------------------------------------------
    # CLASSIFICATION SECTION
    # ----------------------------------------------------------

    # Instead of Flatten()
    GlobalAveragePooling2D(),


    Dense(
        128,

        activation="relu",

        kernel_regularizer=regularizers.l2(
            0.0001
        )
    ),


    BatchNormalization(),


    Dropout(
        0.40
    ),


    Dense(
        5,
        activation="softmax"
    )

])


model.summary()


# ==============================================================
# 12. COMPILE
# ==============================================================

optimizer = keras.optimizers.Adam(
    learning_rate=0.001
)


model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==============================================================
# 13. CALLBACKS
# ==============================================================

early_stopping = keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.000001,

    verbose=1
)


# ==============================================================
# 14. TRAIN MODEL
# ==============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")


history = model.fit(

    train_images,

    train_labels,

    validation_data=(
        validation_images,
        validation_labels
    ),

    epochs=50,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=True,

    verbose=1

)


print(
    "\nTraining finished."
)


# ==============================================================
# 15. TRAINING / VALIDATION ACCURACY
# ==============================================================

training_accuracy = history.history[
    "accuracy"
]


validation_accuracy = history.history[
    "val_accuracy"
]


epochs = range(
    1,
    len(training_accuracy) + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs,
    training_accuracy,
    label="Training Accuracy"
)


plt.plot(
    epochs,
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

plt.show()


# ==============================================================
# 16. TRAINING / VALIDATION LOSS
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
    epochs,
    training_loss,
    label="Training Loss"
)


plt.plot(
    epochs,
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

plt.show()


# ==============================================================
# 17. TEST MODEL
# ==============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")


test_loss, test_accuracy = model.evaluate(

    test_images,

    test_labels,

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
# 18. MAKE PREDICTIONS
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
# 19. CONFUSION MATRIX
# ==============================================================

confusion_matrix = tf.math.confusion_matrix(

    test_labels,

    predicted_labels,

    num_classes=5

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
    range(5),
    class_names,
    rotation=45
)


plt.yticks(
    range(5),
    class_names
)


for i in range(5):

    for j in range(5):

        plt.text(

            j,

            i,

            confusion_matrix[i, j],

            ha="center",

            va="center"

        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ==============================================================
# 20. CORRECT CLASSIFICATION RATE PER CLASS
# ==============================================================

class_accuracy = []


print("\n========================================")
print("CORRECT CLASSIFICATION RATE PER CLASS")
print("========================================")


for i in range(5):

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

plt.show()


# ==============================================================
# 21. 10 EXAMPLE PREDICTIONS
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
        test_images[image_index]
    )


    predicted_class = (
        predicted_labels[
            image_index
        ]
    )


    true_class = (
        test_labels[
            image_index
        ]
    )


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

        fontsize=9

    )


    plt.axis(
        "off"
    )


plt.suptitle(
    "Example Model Predictions"
)


plt.tight_layout()

plt.show()


# ==============================================================
# 22. 30 RANDOM PREDICTIONS
# ==============================================================

random_indices = rng.choice(

    len(test_images),

    30,

    replace=False

)


plt.figure(
    figsize=(15, 18)
)


correct_count = 0


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


    true_label = (
        test_labels[
            image_index
        ]
    )


    predicted_label = (
        predicted_labels[
            image_index
        ]
    )


    confidence = (

        predictions[
            image_index
        ][
            predicted_label
        ]

        * 100

    )


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


plt.show()


sample_accuracy = (
    correct_count / 30
) * 100


print(

    "\nAccuracy among 30 random images:",

    round(
        sample_accuracy,
        2
    ),

    "%"

)


# ==============================================================
# 23. SHOW 30 MISCLASSIFIED IMAGES
# ==============================================================

wrong_indices = np.where(

    predicted_labels
    !=
    test_labels

)[0]


print(

    "\nTotal misclassified test images:",

    len(
        wrong_indices
    )

)


number_to_show = min(

    30,

    len(
        wrong_indices
    )

)


if number_to_show > 0:


    selected_wrong = rng.choice(

        wrong_indices,

        number_to_show,

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


        true_label = (
            test_labels[
                image_index
            ]
        )


        predicted_label = (
            predicted_labels[
                image_index
            ]
        )


        confidence = (

            predictions[
                image_index
            ][
                predicted_label
            ]

            * 100

        )


        plt.title(

            f"Pred: {class_names[predicted_label]}\n"
            f"True: {class_names[true_label]}\n"
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


    plt.show()


# ==============================================================
# 24. FINAL RESULTS
# ==============================================================

best_validation_accuracy = max(
    validation_accuracy
)


best_validation_loss = min(
    validation_loss
)


best_validation_epoch = (
    np.argmax(
        validation_accuracy
    )
    + 1
)


best_loss_epoch = (
    np.argmin(
        validation_loss
    )
    + 1
)


print("\n========================================")
print("FINAL RESULTS")
print("========================================")


print(
    "Epochs actually trained:",
    len(
        training_accuracy
    )
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
    best_validation_epoch
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
    best_loss_epoch
)


print(
    "Testing accuracy:",
    round(
        test_accuracy
        * 100,
        2
    ),
    "%"
)


print(
    "\nProgram finished."
)