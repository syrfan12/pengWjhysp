# penggunaan
# sudo python3 buat_dataset.py --samVid video/asli.mkv --hsDataset dataset/asli --detektor face_detector --bw 1
# sudo python3 buat_dataset.py --samVid video/foto.mkv --hsDataset dataset/foto --detektor face_detector --bw 1

import numpy as np
import argparse
import cv2
import os

# pemisahan input data terminal
ap = argparse.ArgumentParser()
ap.add_argument("-sa", "--samVid", type=str, required=True,
	help="folder sample video 'foto' dan 'asli'")
ap.add_argument("-hs", "--hsDataset", type=str, required=True,
	help="hasil crop untuk dijadikan dataset")
ap.add_argument("-d", "--detektor", type=str, required=True,
	help="folder konfigurasi detektor wajah")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="toleransi minimun deteksi wajah")
ap.add_argument("-bw", "--buangFrame", type=int, default=16,
	help="jumlah frame yang dibuang sebelum data wajah disimpan")
args = vars(ap.parse_args())

# inisialisasi data face detektor
print("menyiapkan data face detektor...")
protoPath = os.path.sep.join([args["detektor"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detektor"],
	"res10_300x300_ssd_iter_140000.caffemodel"]) # model pengenalan wajah yg sudah ditraining 
												 # dari opencv berbasis deep learning
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# memulai video
vs = cv2.VideoCapture(args["samVid"])

read = 0
saved = 0

# memulai perulangan
while True:
	# baca video dan dimasukan ke variable frame
	(grabbed, frame) = vs.read()
	

	# cek pembacaan video berhasil atau tidak, jika tidak program berhenti
	if not grabbed:
		break
	print("ds")

	# penghitung index nama gambar wajah yang akan disimpan
	read += 1

	# cek ada frame yang perlu dilewat sesuai pengaturan skip, jika 1 tidak ada yg di lewat
	if read % args["buangFrame"] != 0:
		continue

	# mencari daerah kasar yang dianggap wajah
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# proses pendeteksian wajah
	net.setInput(blob)
	detections = net.forward()

	# jika ada wajah terdeteksi
	if len(detections) > 0:
		# mengambil area yang dideteksi sebagai wajah dengan probabilitas yang paling besar
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]


			p = os.path.sep.join([args["hsDataset"],
				"{}.png".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] menyimpan {}".format(p))


vs.release()
cv2.destroyAllWindows()
