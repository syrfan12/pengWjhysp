# PENGGUNAAN
# python3 fc-enc.py --dataset datasetRec --encodings encodings.pickle --detection-method hog

# import library yang di perlukan
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Parsing Argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="folder dataset peng wajah")
ap.add_argument("-e", "--encodings", required=True,
	help="nama file hasil encoding")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="metode pengenalan wajah `hog` atau `cnn`")
args = vars(ap.parse_args())

# Ambil gambar dari folder dataset
print("[INFO] mendapatkan model wajah...")
imagePaths = list(paths.list_images(args["dataset"]))

# inisialiassi wajah yang di kenal
knownEncodings = []
knownNames = []

# loop di direktori gambar
for (i, imagePath) in enumerate(imagePaths):
	# Ambil nama dari masing-masing folder
	print("[INFO] Memproses gambar {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# konversi ke RGB (OpenCV ordering) ke dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# deteksi (x, y) koordinat dari kotak wajah
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# Pemrosesan Wajah
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop semua proses encoding
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# menyimpan data encoding dan nama wajah
print("[INFO] Memproses serialize encoding...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
