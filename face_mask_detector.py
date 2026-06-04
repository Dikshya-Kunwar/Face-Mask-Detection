
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings("ignore")

INIT_LR= 1e-4
EPOCHS =10
BS = 20

DIRECTORY = r"./data"
CATEGORIES = ["with_mask","without_mask"]
data=[]
labels=[]
batch_size = 32
img_size = (224, 224)

dataset = tf.keras.utils.image_dataset_from_directory(
    DIRECTORY,
    labels="inferred",
    label_mode="categorical", 
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

class_names = dataset.class_names
dataset = dataset.map(
    lambda x, y: (
        tf.keras.applications.mobilenet_v2.preprocess_input(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

baseModel = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

baseModel.trainable = False 

model = tf.keras.models.Sequential([
    baseModel,
    tf.keras.layers.AveragePooling2D(pool_size=(7,7)),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


opt= tf.keras.optimizers.Adam(learning_rate=INIT_LR,weight_decay=INIT_LR/EPOCHS)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history= model.fit(dataset, epochs=10)


y_true = np.concatenate(
    [y.numpy() for x, y in dataset],
    axis=0
)


y_pred_probs = model.predict(dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

if len(y_true.shape) > 1:
    y_true = np.argmax(y_true, axis=1)

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

model.save("mask_detector.keras")


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig("plot.png")


