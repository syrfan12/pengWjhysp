# PENGGUNAAN
# python3 training_pemb_foto.py --dataset dataset --model pemb.model --hsenc pemb.pickle

import matplotlib
matplotlib.use("Agg")
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Parsing Argumen
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="folder dataset penmbeda asli dan foto")
ap.add_argument("-m", "--model", type=str, required=True,
	help="nama file hasil model yang sudah ditraining")
ap.add_argument("-hs", "--hsenc", type=str, required=True,
	help="nama file encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="plot hasil loss/akurasi")
args = vars(ap.parse_args())

INIT_LR = 1e-4 # learning race training
BS = 8  	   # ukuran batch
EPOCHS = 50    # banyaknya perulangan saat training

print("[INFO] Memuat gambar...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	# mengambil nama folder pemilik wajah dan juga 
	# mengkonversi tiap gambar menjadi berukuran 32x32 pixel
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	# tiap gambar dan nama wajah dimasukan kedalam list
	data.append(image)
	labels.append(label)

# rescaling data untuk mempermudah proses training
data = np.array(data, dtype="float") / 255.0

# merubah representasi pemilik wajah dari nama menjadi urutan angka(0, 1, 2...n)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# membagi semau data yg siap menjadi 75% untuk dipakai menjadi bahan training dan
# 25% untuk digunakan sebagai acuan saat testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# mempersiapkan template gambar agar dapat dilakukan training dengan imagedatagenerator
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# menyiapkan model training
print("[INFO] menyiapkan model training...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# proses training
print("[INFO] proses training untuk epoch(perulangan) ke {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# hasil akhir training
print("[INFO] melakukan prediksi model...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# menyimpan model
print("[INFO] menyimpan model di '{}'...".format(args["model"]))
model.save(args["model"])

# menyimpan hasil encoder
f = open(args["hsenc"], "wb")
f.write(pickle.dumps(le))
f.close()

# plot hasil training loss dan akurasi
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("training loss dan akurasi")
plt.xlabel("Epoch(perulangan) #")
plt.ylabel("Loss/Akurasi")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
