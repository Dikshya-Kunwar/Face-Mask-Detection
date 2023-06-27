import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


INIT_LR= 1e-4
EPOCHS =10
BS = 20

DIRECTORY = r"C:\python\FaceMaskDetection\Dataset\Test"
CATEGORIES = ["WithMask","WithoutMask"]

data=[]
labels=[]

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image =tf.keras.utils.load_img(img_path,target_size=(224,224))
        image = tf.keras.utils.img_to_array(image)
        image= tf.keras.applications.mobilenet_v2.preprocess_input(image)
        data.append(image)
        labels.append(category)

lb=LabelBinarizer()
labels = lb.fit_transform(labels)
labels= tf.keras.utils.to_categorical(labels)

data = np.array(data,dtype="float32")
labels=np.array(labels)

(trainX,testX,trainY,testY)= train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=42)


model = tf.keras.models.Sequential(tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,input_tensor=tf.keras.layers.Input(shape=(224,224,3))))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(7,7)))
model.add(tf.keras.layers.Flatten(name="flatten"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2,activation="softmax"))


opt= tf.keras.optimizers.Adam(learning_rate=INIT_LR,weight_decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

H = model.fit(
	trainX, trainY, batch_size=BS,
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

prediction = model.predict(testX, batch_size=BS)
prediction = np.argmax(prediction,axis=1)

print(classification_report(testY.argmax(axis=1),prediction,target_names=lb.classes_))

model.save("mask_detector.model",save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig("plot.png")